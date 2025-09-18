# config for finetuning GPT-2 XL on custom mydataset
# launch as: python train.py config/finetune_mydataset.py

out_dir = 'out-finetune-mydataset'
eval_interval = 100
log_interval = 1
eval_iters = 20
always_save_checkpoint = True
init_from = 'gpt2'

# data
dataset = 'mydataset'
batch_size = 4
block_size = 512
gradient_accumulation_steps = 2

# model
dropout = 0.1

# adamw optimizer
learning_rate = 1e-5  # lower learning rate for finetuning
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 50
lr_decay_iters = 5000
min_lr = 1e-6

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False

# wandb logging
wandb_log = False
