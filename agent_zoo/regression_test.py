import gym, roboschool, time, tqdm, resource
import numpy as np

#
# Run this script to check if your Bullet version is good.
#

help = """
Expected values, score 2370 ± 770, speed and memory:
Linux   i7 4.0 GHz:  1060 FPS   80 Mb
Macbook i7 3.0 GHz:  1400 FPS   64 Mb
(not a mistake, Bullet is faster on mac as of May 2017)
"""

from RoboschoolHumanoidFlagrunHarder_v0_2017may import SmallReactivePolicy as HumanoidPolicy

def bullet_regression_test_score_and_fps(env):
    pi = HumanoidPolicy(env.observation_space, env.action_space)
    t1 = time.time()
    frame = 0
    scores_array = []
    for episode in tqdm.tqdm(range(100)):
        score = 0
        obs = env.reset()
        while 1:
            a = pi.act(obs)
            obs, r, done, _ = env.step(a)
            score += r
            frame += 1
            if frame==1000: break
            if done: break
        scores_array.append(score)
    t2 = time.time()
    scores_array = np.array(scores_array)
    print("Score: %0.1f ± %0.1f" % (scores_array.mean(), scores_array.std()))
    print("Speed: %0.1f FPS (%i frames in %0.1fs)" % (frame/(t2-t1), frame, t2-t1))

def bullet_regression_test_memory_leak(env):
    print("Memory used: %0.1f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    print("Now do 5000 resets:")
    for episode in tqdm.tqdm(range(5000)):
        env.reset()
    print("Memory used: %0.2f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

if __name__=="__main__":
    env = gym.make("RoboschoolHumanoidFlagrunHarder-v0")
    env.reset()
    bullet_regression_test_score_and_fps(env)
    bullet_regression_test_memory_leak(env)
    print(help)
