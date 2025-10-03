from attrs import define

from observer.reward_epoch_manager import SigningPolicy


@define
class SigningPolicyManager:
    current_policy: SigningPolicy
    previous_policy: SigningPolicy
