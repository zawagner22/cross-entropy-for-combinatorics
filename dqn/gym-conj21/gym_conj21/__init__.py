from gym.envs.registration import register

register(
    id='conj21-v0',
    entry_point='gym_conj21.envs:Conj21Env',
)