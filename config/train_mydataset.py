# config for training GPT on custom mydataset
# launch as: python train.py config/train_mydataset.py

out_dir = 'out-mydataset'
eval_interval = 100
log_interval = 1
eval_iters = 20
always_save_checkpoint = True
init_from = 'scratch'

# data
dataset = 'mydataset'
batch_size = 8
block_size = 512
gradient_accumulation_steps = 1

# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False

# adamw optimizer
learning_rate = 3e-4
max_iters = 10000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = 10000
min_lr = 3e-5

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
