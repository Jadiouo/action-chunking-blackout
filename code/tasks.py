"""Task registry (P1). Default = transfer_cube, i.e. exactly the original setup of T1-T4.
Both tasks: ACT episode_len 400 (lerobot AlohaEnv config / original ACT sim constants), success = max reward 4,
t0 ~ U{50..250} (unchanged: the insertion episode is also 400 steps long)."""
TASKS = {
    "transfer_cube": dict(repo="lerobot/act_aloha_sim_transfer_cube_human", env_id="gym_aloha/AlohaTransferCube-v0",
                          max_steps=400, t0_lo=50, t0_hi=250, success_reward=4),
    "insertion": dict(repo="lerobot/act_aloha_sim_insertion_human", env_id="gym_aloha/AlohaInsertion-v0",
                      max_steps=400, t0_lo=50, t0_hi=250, success_reward=4),
}
