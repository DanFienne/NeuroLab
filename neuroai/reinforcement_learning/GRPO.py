#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch


class GRPO(object):
    def __init__(self, n_states, n_actions):
        self.ref_model = "DeepSeek-R1"
        self.beta = 0.04
        pass

    def prepare_inputs(self, inputs, rewards):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logit_to_keep = completion_ids.size(1)
        ref_per_token_log_prob = self.get_per_token_log_prob(self.ref_model, input_ids, logit_to_keep)
        mean_grouped_rewards = rewards.mean(dim=1)
        std_grouped_rewards = rewards.std(dim=1)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        result = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_log_prob": ref_per_token_log_prob,
            "advantages": advantages,
        }
        return result

    @staticmethod
    # Get the per-token log probabilities for the completions for the model and the reference model
    def get_per_token_log_prob(model, input_ids, logit_to_keep):
        logit = model().logits # (b, len, vocab)
        # exclude the last logit <EOS>
        logit = logit[:, :-1, :] # (b, len-1, vocab)

        per_token_log_prob = []
        for logit_row, input_ids_row in zip(logit, input_ids[:, -logit_to_keep:]):
            # log_probs: logP(w_i|w_1,w_2...)
            log_probs = logit_row.log_softmax(dim=-1)
            # token_log_prob: get each token log prob, like list index
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_log_prob.append(token_log_prob)
        return torch.stack(per_token_log_prob)

    def compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        # completion_mask only choice before eos token
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logit_to_keep = completion_ids.size(1)

        per_token_log_prob = self.get_per_token_log_prob(model, input_ids, logit_to_keep)

        # compute KL divergence between the model and the reference model
        ref_per_token_log_prob = inputs["ref_per_token_log_prob"]
        per_token_kl = (torch.exp(ref_per_token_log_prob - per_token_log_prob)
                        - (ref_per_token_log_prob - per_token_log_prob) - 1)

        advantages = inputs["advantages"]
        # assume 𝜋𝜃𝑜𝑙𝑑 = 𝜋𝜃 for simplified analysis
        per_token_loss = torch.exp(per_token_log_prob - per_token_log_prob.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        return loss

    def predict(self, inputs, model, rewards):
        inputs = self.prepare_inputs(inputs, rewards)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None
