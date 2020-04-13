
- [ ] Create Episode class with agent, max_steps, uid, replay?
- [ ] Remove agent from EpisodePool, rename to EnvPool
- [ ] Add episode queue to EnvPool
    - [ ] def enqueue
    - [ ] def dequeue
    - [ ] Or use built-in queue class? and @property?
    - [ ] Add 'QUEUE_COMPLETED' event, 'QUEUE_EMPTY' event?
- [ ] train mode binds to QUEUE_EMPTY and enqueues another episode (unless n_steps > threshold?)
- [ ] enjoy mode enqueues N episodes, listens for QUEUE_COMPLETED event or calls .join() on EnvPool?
- [ ] Eval mode could call enjoy mode then compute metrics?  Or just run the same code then compute metrics at the end (and write to disk).
- [ ] Eval 
- [ ] Enjoy mode creates agent, episodepool, runs.
- [ ] Eval mode creates agent, episodepool, collects replay buffer, computes metrics, saves metrics?  Theoretically with checkpoint we could re-analyze if metrics are updated.
- [ ] Train mode... let's get here!
- [ ] Simplify Agent <-> Env interaction so that we don't need to over-engineer the obs space
- [ ] Remove lab_mode from os.environ
- [ ] Change eval and enjoy mode to only specify checkpoint
    - [ ] Remove spec_util.get_eval_spec
    - [ ] Store model checkpoint paths in output specfile so we don't need to assume them?
- [ ] Broke calls to env.render() (was handled in envs/openai.py and envs/unity.py)
- [ ] Broke other env settings handled by envs/base.py and envs/openai.py and envs/unity.py
- [ ] Create Episode class to manage episode, events, storage?
- [ ] Hook EnvPool up to use Episode class
- [ ] act() possibly needs to receive environment UIDs corresponding to the batch.  Should we use environment IDs or episode IDs?  If we use episode IDs, it is "safer".  I can't think of a reason why it would be good for the agent to know environments apart by ID.
- [ ] Implement train loop & train spec
- [ ] Implement Agent interface?
- [ ] Cleanup / get rid of vec_env, envs/base, envs/unity, envs/openai?

Ideas
- Agent only has act(o,uids) and optional on_step(oaordi), on_episode_start(uid), and on_episode_terminated(uid, done) and built-in train property and abstract save and load which receives a directory path in which to save or load checkpoints. Also consider should_start_episode() for control over 
- Don't forget about being able to reset() an EpisodePool after an epoch, OR pushing the epoch length higher into the programming model, i.e. into the spec.
- Agents are extensible or copy/pasteable 
- Options for episode mgmt
    - for _ in range(10): pool.schedule_episode() or pools.schedule_episodes(10)
    - pool.register_hook('after_episode', lambda: self.pool.cancel() if self.n_episodes++ > self.max_episodes)
    - ep = Episode(agent, max_t=10); pool.enqueue(ep);  pool.mode = 'manual'.
    - pool.allow_new_episodes = False
    - The crux of the issue is that you don't want to bias toward shorter episodes, which can be more or less rewarding depending on the env.  This is especially so for evaluation.  You *may* want to evaluate in parallel, but you need a mechanism for taking 10 episodes, AND having these be the first 10 episodes started, not the 10 fastest-finishing episodes.  Another issue is that we actually can't just stop new episodes in an EpisodePool because we need the pool to output a certain number of values, right?  Obs should be a certain lenght?  Actually NNs usually have flexible batch size...  So maybe a non-issue or minor issue. Damn. :) Ok, so that's fine.  In general, is it safe to assume that partial episodes are fairly useless?  Hmm... I don't think it's necessarily true, BUT finished episodes are probably more useful. And finished episodes are guaranteed to be useful.  In task-free learning, I suppose all timesteps are useful.  Hmm....  Also in task-free learning, there may not be a clear end to the episode.  So we definitely shouldn't build the API around episodes and the assumption that they will finish.  We do need them for reward-based envs.  We can't evaluate on a fixed # of steps.  For task-free agents or reward-free envs... wtf does it mean to evaluate?  
- Agent should also be async so it can perform other work while the envs are operating.  Is this the difference between train and eval?  What else would the agent do besides 
    

Ideas while thinking about vec_env:
- Always make an EnvPool (with num_envs=1 and/or concurrency=None for simple use cases and debugging)
- EnvPool is combined with agent in EpochRunner?  Where an epoch is a certain number of episodes or rollout timesteps?  And this runs the agent with the EnvPool?
  - Agent always receives list of observations and returns list of actions to make interface straight-forward
  - EpochRunner should have option (or agent should have config) for receiving scalar / non-list observations and returning non-list actions.
  - Or... we provide Agent base class to extend, and if it's extended, we pass list obs and expect list actions. If not, we expect scalars?  Or the base class can 
- Is there a case for having an Episode class where the Agent can store metadata or reference in its metadata?  The idea is that if you cancel/abort the episode, then agent-specific episode data could be garbage-collected, e.g. episodic latent memory buffers.  Is this easily accomplished with an after_episode callback on the agent, and let it store everything it needs for the episode?  Or should it attach references to the Episode itself, which can be garbage-collected without the agent knowing (or needing to know) about it?  This is a nice idea! This would make even agents with latent memory more "stateless", or that the stateful properties should go on the episode (or on the body which is incarnated for the episode).  And this helps distinguish whether data should belong in the agent (if it is stateless, such as neural networks, world models, and causal graphs) or in the body (latent episodic memory and other model inputs which vary based on the episode)
- Then the agent is still environment-dependent... it's the *learning algorithm* which is environment-agnostic, the learned model (agent) is environment-specific but episode-agnostic, and the body is both environment- and episode-specific.
- interface for Agent.get_initial_episode_state returns None by default
- agent.act receives observations, episode_states if episode_states are present, otherwise just observations.
- Should Agent be notified for 
- How to implement episodic memory?  Should it be "on" by default?  Should you return a Memory object from get_initial_episode_state?  Should you implement a Body class which handles events like "steps" however it pleases, and has an __init__ function for initialization?  How to handle death / end of episode?  Should we instead append to a replay buffer every time without discrimination for terminated episodes?  I suppose this makes sense — in the real world, if an agent "died", you wouldn't be able to communicate with it or post-process it.  So communication of experiences needs to be ongoing.  This will help bridge between dev and production.  The agent/body design could also be interesting for developing agents "at the edge" which have limited communication capabilities with the central (intelligent) agent.  However, this could easily be accomplished in development by storing an entire network in the "body" as a stateful component. Then the agent.act() function should self-enforce that it only accesses information within the body.  So, this is possible, but not super efficient.  Maybe by default the body should be empty and should call the central agent.act() function for you?  And only if the body overrides act() will it be able to operate "at the edge"?
- Ok, so in the simple case, one env DOES represent one body.  In the real world, the "env" is always the same (the real world) but the body changes e.g. if you embed the agent in different robots or "bodies" which all exist in the same world (the real universe).  BUT in a multi-agent environment with a variable number of agents, then each agent represents one "body" where again the "env" is unchanging.  So you could either attach to this one agent which controls all bodies (e.g. using a shared memory across bodies) or multi-agent with one agent per body.  You could even have multiple agents for one body.  So... do we really need to provide mechanisms for managing this?  Is there a case where the central agent can't decide how to control each body (possibly "at the edge"), e.g. by assigning one brain to all of them, or 