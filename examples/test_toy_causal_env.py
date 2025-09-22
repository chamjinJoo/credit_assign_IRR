#!/usr/bin/env python3
"""
ToyCausalEnv Test Script

This script tests the basic functionality of ToyCausalEnv and 
verifies observation and reward structure.
"""

import os
# Resolve OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from pci_mappo.envs.multi_agent_env import ToyCausalEnv


def test_basic_functionality():
    """Basic functionality test"""
    print("=" * 50)
    print("ToyCausalEnv Basic Functionality Test")
    print("=" * 50)
    
    env = ToyCausalEnv(num_agents=2, episode_length=5)
    
    print(f"Number of agents: {env.num_agents}")
    print(f"Action dimension: {env.get_action_dim()}")
    print(f"Observation dimension: {env.get_obs_dim()}")
    print(f"Episode length: {env.episode_length}")
    print()
    
    # Initialize environment
    obs = env.reset()
    print("Initial observations:")
    for agent_id, observation in obs.items():
        print(f"  Agent {agent_id}: {observation}")
        print(f"    High reward signal: {observation[0]}")
        print(f"    Reward location: {observation[1]} ({'Left' if observation[1] == -1 else 'Right'})")
    print()
    
    return env


def test_random_policy(env, num_episodes=3):
    """Test environment with random policy"""
    print("=" * 50)
    print("Random Policy Test")
    print("=" * 50)
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}:")
        print("-" * 30)
        
        obs = env.reset()
        env.render()
        
        total_rewards = {agent_id: 0 for agent_id in env.agents}
        
        for step in range(env.episode_length):
            print(f"\nStep {step + 1}:")
            
            # Random action selection
            actions = {}
            for agent_id in env.agents:
                actions[agent_id] = np.random.randint(0, 2)  # 0: Left, 1: Right
            
            print(f"Actions: {actions}")
            for agent_id, action in actions.items():
                action_str = "Left" if action == 0 else "Right"
                print(f"  Agent {agent_id}: {action_str}")
            
            # Execute step
            obs, rewards, dones, info = env.step(actions)
            
            print(f"Rewards: {rewards}")
            for agent_id, reward in rewards.items():
                total_rewards[agent_id] += reward
            
            env.render()
            
            if all(dones.values()):
                break
        
        print(f"\nEpisode {episode + 1} Total Rewards:")
        for agent_id, total_reward in total_rewards.items():
            print(f"  Agent {agent_id}: {total_reward}")


def test_optimal_policy(env, num_episodes=3):
    """Test environment with optimal policy (following reward locations)"""
    print("=" * 50)
    print("Optimal Policy Test (Following Reward Locations)")
    print("=" * 50)
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}:")
        print("-" * 30)
        
        obs = env.reset()
        env.render()
        
        total_rewards = {agent_id: 0 for agent_id in env.agents}
        
        for step in range(env.episode_length):
            print(f"\nStep {step + 1}:")
            
            # Optimal action selection (based on reward location)
            actions = {}
            for agent_id in env.agents:
                reward_location = obs[agent_id][1]  # 2nd element of observation
                if reward_location == -1:  # Reward on left
                    actions[agent_id] = 0  # Move left
                else:  # Reward on right
                    actions[agent_id] = 1  # Move right
            
            print(f"Actions: {actions}")
            for agent_id, action in actions.items():
                reward_loc = obs[agent_id][1]
                action_str = "Left" if action == 0 else "Right"
                reward_str = "Left" if reward_loc == -1 else "Right"
                print(f"  Agent {agent_id}: {action_str} (Reward location: {reward_str})")
            
            # Execute step
            obs, rewards, dones, info = env.step(actions)
            
            print(f"Rewards: {rewards}")
            for agent_id, reward in rewards.items():
                total_rewards[agent_id] += reward
            
            env.render()
            
            if all(dones.values()):
                break
        
        print(f"\nEpisode {episode + 1} Total Rewards:")
        for agent_id, total_reward in total_rewards.items():
            print(f"  Agent {agent_id}: {total_reward}")


def test_action_success_probability():
    """Test action success probability"""
    print("=" * 50)
    print("Action Success Probability Test")
    print("=" * 50)
    
    env = ToyCausalEnv(num_agents=2, episode_length=1)
    
    num_trials = 1000
    success_count = 0
    
    for trial in range(num_trials):
        obs = env.reset()
        
        # Agent tries to move to reward location
        reward_location = obs[0][1]
        intended_action = 1 if reward_location == 1 else 0  # Action matching reward location
        
        actions = {0: intended_action, 1: np.random.randint(0, 2)}
        obs, rewards, dones, info = env.step(actions)
        
        # If reward received, agent successfully moved in intended direction
        if rewards[0] > 0:
            success_count += 1
    
    success_rate = success_count / num_trials
    print(f"Total trials: {num_trials}")
    print(f"Successful attempts: {success_count}")
    print(f"Success rate: {success_rate:.3f}")
    print(f"Expected success rate: 0.900")
    print(f"Difference: {abs(success_rate - 0.9):.3f}")


def test_high_reward_events():
    """Test high reward events"""
    print("=" * 50)
    print("High Reward Events Test")
    print("=" * 50)
    
    # Run multiple episodes to check high reward event occurrence
    env = ToyCausalEnv(num_agents=2, episode_length=5)
    
    high_reward_count = 0
    total_episodes = 50
    
    for episode in range(total_episodes):
        obs = env.reset()
        
        # Check if this episode has high reward events
        has_high_reward = any(event != 0 for event in env.reward_events)
        if has_high_reward:
            high_reward_count += 1
            print(f"Episode {episode + 1}: High reward event occurred")
            for step, event in enumerate(env.reward_events):
                if event == 1:
                    print(f"  Step {step}: Agent 1's high reward signal to agent 2")
                elif event == 2:
                    print(f"  Step {step}: Agent 2's high reward signal to agent 1")
    
    print(f"\nTotal episodes: {total_episodes}")
    print(f"Episodes with high reward events: {high_reward_count}")
    print(f"Occurrence rate: {high_reward_count / total_episodes:.3f}")


def plot_reward_distribution():
    """Visualize reward distribution"""
    print("=" * 50)
    print("Reward Distribution Analysis")
    print("=" * 50)
    
    env = ToyCausalEnv(num_agents=2, episode_length=5)
    
    # Compare random policy and optimal policy
    random_rewards = []
    optimal_rewards = []
    
    num_episodes = 100
    
    # Random policy
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(env.episode_length):
            actions = {agent_id: np.random.randint(0, 2) for agent_id in env.agents}
            obs, rewards, dones, info = env.step(actions)
            episode_reward += sum(rewards.values())
        
        # Calculate average reward per agent per step
        average_reward = episode_reward / (env.num_agents * env.episode_length)
        random_rewards.append(average_reward)
    
    # Optimal policy
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(env.episode_length):
            actions = {}
            for agent_id in env.agents:
                reward_location = obs[agent_id][1]
                actions[agent_id] = 0 if reward_location == -1 else 1
            
            obs, rewards, dones, info = env.step(actions)
            episode_reward += sum(rewards.values())
        # Calculate average reward per agent per step
        average_reward = episode_reward / (env.num_agents * env.episode_length)
        optimal_rewards.append(average_reward)
    
    # Print results
    print(f"Random policy average reward: {np.mean(random_rewards):.3f} ± {np.std(random_rewards):.3f}")
    print(f"Optimal policy average reward: {np.mean(optimal_rewards):.3f} ± {np.std(optimal_rewards):.3f}")
    
    # Create plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(random_rewards, bins=20, alpha=0.7, label='Random Policy')
    plt.hist(optimal_rewards, bins=20, alpha=0.7, label='Optimal Policy')
    plt.xlabel('Average Reward per Agent per Step')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution Comparison')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.boxplot([random_rewards, optimal_rewards], tick_labels=['Random', 'Optimal'])
    plt.ylabel('Average Reward per Agent per Step')
    plt.title('Reward Distribution Box Plot')
    
    plt.tight_layout()
    plt.savefig('toy_causal_env_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return random_rewards, optimal_rewards


def main():
    """Main test function"""
    print("ToyCausalEnv Comprehensive Test Started")
    print("=" * 60)
    
    # Basic functionality test
    env = test_basic_functionality()
    
    # Random policy test
    test_random_policy(env, num_episodes=2)
    
    # Optimal policy test
    test_optimal_policy(env, num_episodes=2)
    
    # Action success probability test
    test_action_success_probability()
    
    # High reward events test
    test_high_reward_events()
    
    # Reward distribution analysis
    try:
        plot_reward_distribution()
        print("\nReward distribution graph saved as 'toy_causal_env_results.png'.")
    except ImportError:
        print("\nmatplotlib is not installed, cannot generate graphs.")
        print("Install with 'pip install matplotlib' to view graphs.")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")


if __name__ == "__main__":
    main()
