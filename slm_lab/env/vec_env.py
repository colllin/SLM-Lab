# Wrappers for parallel vector environments.
# Adapted from OpenAI Baselines (MIT) https://github.com/openai/baselines/tree/master/baselines/common/vec_env


# class CloudpickleWrapper(object):
#     '''
#     Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
#     '''
# 
#     def __init__(self, x):
#         self.x = x
# 
#     def __getstate__(self):
#         import cloudpickle
#         return cloudpickle.dumps(self.x)
# 
#     def __setstate__(self, ob):
#         import pickle
#         self.x = pickle.loads(ob)




