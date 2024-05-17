import os
import sys
from datetime import datetime


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy


if __name__ == "__main__":
    out_csv = f"outputs/mini/2way-single-intersection/ql"

    env = SumoEnvironment(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file='sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
        out_csv_name=out_csv,
        use_gui=True,
        num_seconds=800000,
        min_green=10,
        max_green=50,
        sumo_warnings=False,
    )

    for run in range(1, 2):
        initial_states = env.reset()
        ql_agents = {
            ts: QLAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.action_space,
                alpha=.1,
                gamma=.99,
                exploration_strategy=EpsilonGreedy(
                    initial_epsilon=.05, min_epsilon=0.005, decay=1.0
                ),
            )
            for ts in env.ts_ids
        }

        done = {"__all__": False}
        
        while not done["__all__"]:
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

            s, r, done, _ = env.step(action=actions)

            for agent_id in ql_agents.keys():
                ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])
        env.save_csv(out_csv, run)
        env.close()
