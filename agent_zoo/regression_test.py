import gym, roboschool, time, tqdm, resource
import numpy as np

#
# Run this script to check if your Bullet version is good.
#

help = """
Expected values:
Linux  i7 4.0 GHz:   2400 ± 600   1060 FPS   80M
OS X   i7 2.2 GHz:    -/-
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
    print("Memory used: %0.2fM" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    print("Now do 5000 resets:")
    for episode in tqdm.tqdm(range(5000)):
        env.reset()
    print("Memory used: %0.2fM" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

if __name__=="__main__":
    env = gym.make("RoboschoolHumanoidFlagrunHarder-v0")
    env.reset()
    bullet_regression_test_score_and_fps(env)
    bullet_regression_test_memory_leak(env)
    print(help)
