import string
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import wandb
# TODO allow setting run name with command-line arg
wandb.init(project="rl_repl")
config = wandb.config

config.EXPERIMENT_NAME = 'master'

# Hyperparameters ##############
config.MAX_INPUT_OUTPUT_LENGTH = 7
config.MAX_STATE_LEN = config.MAX_INPUT_OUTPUT_LENGTH + 1 + config.MAX_INPUT_OUTPUT_LENGTH + 1

## PPO
config.ACTION_DIM = len(string.digits + '+-*/')
config.LOG_INTERVAL = 20                                            # print avg reward in the interval
config.PPO_BATCH_SIZE = 256                                         # number of episodes ran in the same batch during PPO
config.PPO_N_TRAINING_EPISODES = 6000  # 11                         # max number of PPO training episodes
config.MAX_EPISODES_TIMESTEPS = config.MAX_INPUT_OUTPUT_LENGTH - 1  # max timesteps in each episode
config.UPDATE_EVERY_N_EPISODES = 10                                 # update policy every n episodes
config.MIN_TRAINING_SAMPLE_REWARD = 1                               # Use PPO states with a reward above X to train Seq2Seq
config.PPO_LEARNING_RATE = 0.01
config.BETAS = (0.9, 0.999)                                         # Adam optimizer parameter, provides coefficients used to compute running averages of gradient and its square
config.GAMMA = 0.99                                                 # discount factor
config.K_EPOCHS = 4                                                 # update policy for K epochs
config.EPS_CLIP = 0.2                                               # clip parameter for PPO, decides how big a change in policy we can have
config.CRITIC_DISCOUNT = 0.5                                        # How much to discount the mean square error of the Critic (also known as Value funciton)
config.ENTROPY_BETA = 0.01                                          # Coefficient on the entropy of the actor's prob(action|state), decides how much we want the model to experiment with different policies
config.N_PREVIOUS_LOSSES = 10                                       # To get the reward, compare current translation loss with the average loss of N previous losses
config.LOSS_SMOOTHING_WEIGHT = 0.5                                  # Smooth the seq2seq losses against previous ones using this 
config.PRE_TRAIN_UPDATES = 0                                        # Number of PPO updates to run before updating seq2seq as well
### Actor
config.PPO_ACTOR_HIDDEN_SIZE = 20
config.PPO_ACTOR_LAYER_COUNT = 1
### Critic
config.PPO_CRITIC_HIDDEN_SIZE = 20
config.PPO_CRITIC_LAYER_COUNT = 1

## Seq2seq
config.SEQ2SEQ_PRE_TRAIN_STEPS = 0                                  # number of pre-train (batch) steps for seq2seq model
config.SEQ2SEQ_PRE_TRAINED = None
config.SEQ2SEQ_BATCH_SIZE = 256                                     # batch size in pre-training & testing for seq2seq model
config.MIN_PPO_LOSS = 0.2                                           # only use PPO samples that gave a loss greator than this
config.SEQ2SEQ_EMBEDDING_SIZE = 200                                 # embedding dimension
config.SEQ2SEQ_HIDDEN_SIZE = 200                                    # the dimension of the feedforward network model
config.SEQ2SEQ_NUM_LAYERS = 2                                       # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
config.SEQ2SEQ_NUM_HEADS = 2                                        # the number of heads in the multiheadattention models
config.SEQ2SEQ_DROPOUT = 0.2                                        # the dropout value
config.SEQ2SEQ_LEARNING_RATE = 4.0
config.SEQ2SEQ_MAX_GRADIENT_NORM = 0.5
# End Hyperparameters ##############

# Log/Save Config
config.TOP_K_CHAR_COUNTS = 2                                        # Sums the top K counts in code & output batches
config.NUM_OF_SAMPLES_PER_LOG = 10                                  # Number of sample code-output pairs to store in each log
config.LOG_EVERY_N_EPISODES = 10
config.SAVE_EVERY_N_EPISODES = 100
config.RENDER_EVERY_N_EPISODES = 5
LOGS_FOLDER = 'logs/'
MODELS_FOLDER = 'models/'

#
config.SEQ2SEQ_LOAD_FROM = 'i10000'
config.RENDER = False
RENDER_FILE = LOGS_FOLDER + 'train_render.txt'
