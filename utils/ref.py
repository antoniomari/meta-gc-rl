def evaluate(
    agent,
    env,
    task_id=None,
    config=None,
    finetune_config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    train_dataset=None,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        finetune_config: Configuration dictionary specific to finetuning.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):

        EVAL_INTERVAL = 100
        done = False
        step = 0
        render = []
        old_train_state, old_config = None, None
        while not done:
            if step % EVAL_INTERVAL == 0:

                if old_train_state is not None:
                    agent = agent.replace(network=old_train_state, config=old_config)

                old_config = agent.config
                new_config = agent.config.unfreeze()
                if finetune_config.actor_loss is not None:
                    new_config['actor_loss'] = finetune_config.actor_loss
                if finetune_config.alpha is not None:
                    new_config['alpha'] = finetune_config.alpha
                new_config = old_config.__class__(new_config)
                old_train_state = copy.deepcopy(agent.network)
                opt_state = agent.network.opt_state
                finetune_tx = optax.adam(learning_rate=finetune_config.lr)
                agent = agent.replace(network=agent.network.replace(tx=finetune_tx, opt_state=opt_state), config=new_config)

                finetune_stats = defaultdict(list)
                if finetune_config.num_steps:
                    _filter = train_dataset.prepare_active_sample(agent, observation, goal, finetune_config)
                    num_steps = finetune_config.num_steps if _filter.sum() else 0
                    for _ in range(num_steps):
                        batch = train_dataset.active_sample(finetune_config.batch_size, _filter, goal, finetune_config.ratio, finetune_config.fix_actor_goal)
                        agent, info = agent.update(batch)
                        add_to(finetune_stats, flatten(info))
                actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))

            action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
            action = np.array(action)
            if not config.get('discrete'):
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    stats.update({'finetune/' + k: v for k, v in finetune_stats.items()})
    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders




def filter_from_state_goal(dataset, obs, goal, quantile, slack, state_to_goal_dist=None):

    _obs = dataset['observations']
    ep_id = dataset['terminals'].cumsum() // 2
    ep_id[1:] = ep_id[:-1]
    ep_lens = np.unique(ep_id, return_counts= True)[1]
    assert len(set(ep_lens)) == 1, "All episodes need to have the same length."
    ep_len = ep_lens[0].item()

    mask = np.zeros_like(ep_id)
    _obs = _obs.reshape(-1, ep_len, _obs.shape[-1])
    start_matches = jnp.sqrt(((_obs[..., :2] - obs[:2])**2).sum(-1)) < 1.0

    if state_to_goal_dist is not None:
        LOOK_AHEAD = 10  # minimum length for selected trajectories - avoids trivial solutions
        shift_start_matches = np.zeros_like(start_matches)
        shift_start_matches[:, LOOK_AHEAD:] = start_matches[:, :-LOOK_AHEAD]
        scores = ((shift_start_matches.cumsum(-1) > 0) * state_to_goal_dist.reshape(start_matches.shape))
        scores = np.where(scores==0, scores.max(), scores)
        ep_idxs = np.argsort(scores.min(-1))[:10]
        mask = mask.reshape(-1, ep_len)
        mask[ep_idxs] = 1.
        mask *= (start_matches.cumsum(-1) > 0)  # only keep from matches
        col_indices = np.arange(ep_len)
        mask *= col_indices[None] < scores.argmin(-1)[..., None]  # discard after best point
        return mask.flatten()