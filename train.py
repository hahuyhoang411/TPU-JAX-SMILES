# Jax
import jax
import optax
import flax
import jax.numpy as jnp
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
import jax.tools.colab_tpu
from pathlib import Path

# Model
from transformers import AutoConfig, RobertaConfig

# Tokenizer
import collections
import codecs
import unicodedata
from typing import List, Optional
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from SmilesPE.tokenizer import SPE_Tokenizer

# Helper
import numpy as np
from tqdm import tqdm
import logging
import os
import re

# WandB
import wandb

# Huggingface hub
from huggingface_hub import Repository, create_repo
import shutil

# Remember to set these up
YOUR_TOKEN_HF = ""
DATA_TOKEN_HF = ""
# Remember to set project name and entity name
YOUR_TOKEN_WANDB = ""
YOUR_PROJECT_NAME = ""

jax.local_devices()

import transformers
print(transformers.__version__)

# Download pretrained tokenizer
# !wget https://raw.githubusercontent.com/XinhaoLi74/SmilesPE/master/SPE_ChEMBL.txt

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

class SMILES_SPE_Tokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        spe_file,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'.".format(vocab_file)
            )
        if not os.path.isfile(spe_file):
            raise ValueError(
                "Can't find a SPE vocabulary file at path '{}'.".format(spe_file)
            )
        self.vocab = load_vocab(vocab_file)
        self.spe_vocab = codecs.open(spe_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.spe_tokenizer = SPE_Tokenizer(self.spe_vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        return self.spe_tokenizer.tokenize(text).split(' ')

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, vocab_path):
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

# some default tokens from huggingface
default_toks = ['[PAD]',
                '[unused1]', '[unused2]', '[unused3]', '[unused4]','[unused5]', '[unused6]', '[unused7]', '[unused8]', '[unused9]', '[unused10]',
                '[UNK]', '[CLS]', '[SEP]', '[MASK]']

# atom-level tokens used for trained the spe vocabulary
atom_toks = ['[c-]', '[SeH]', '[N]', '[C@@]', '[Te]', '[OH+]', 'n', '[AsH]', '[B]', 'b',
             '[S@@]', 'o', ')', '[NH+]', '[SH]', 'O', 'I', '[C@]', '-', '[As+]', '[Cl+2]',
             '[P+]', '[o+]', '[C]', '[C@H]', '[CH2]', '\\', 'P', '[O-]', '[NH-]', '[S@@+]',
             '[te]', '[s+]', 's', '[B-]', 'B', 'F', '=', '[te+]', '[H]', '[C@@H]', '[Na]',
             '[Si]', '[CH2-]', '[S@+]', 'C', '[se+]', '[cH-]', '6', 'N', '[IH2]', '[As]',
             '[Si@]', '[BH3-]', '[Se]', 'Br', '[C+]', '[I+3]', '[b-]', '[P@+]', '[SH2]', '[I+2]',
             '%11', '[Ag-3]', '[O]', '9', 'c', '[N-]', '[BH-]', '4', '[N@+]', '[SiH]', '[Cl+3]', '#',
             '(', '[O+]', '[S-]', '[Br+2]', '[nH]', '[N+]', '[n-]', '3', '[Se+]', '[P@@]', '[Zn]', '2',
             '[NH2+]', '%10', '[SiH2]', '[nH+]', '[Si@@]', '[P@@+]', '/', '1', '[c+]', '[S@]', '[S+]',
             '[SH+]', '[B@@-]', '8', '[B@-]', '[C-]', '7', '[P@]', '[se]', 'S', '[n+]', '[PH]', '[I+]',
             '5', 'p', '[BH2-]', '[N@@+]', '[CH]', 'Cl']

# spe tokens
with open('SPE_ChEMBL.txt', "r") as ins:
    spe_toks = []
    for line in ins:
        spe_toks.append(line.split('\n')[0])

spe_tokens = []
for s in spe_toks:
    spe_tokens.append(''.join(s.split(' ')))
print('Number of SMILES:', len(spe_toks))

spe_vocab = default_toks + atom_toks + spe_tokens

with open('vocab_spe.txt', 'w') as f:
    for voc in spe_vocab:
        f.write(f'{voc}\n')

tokenizer = SMILES_SPE_Tokenizer(vocab_file='vocab_spe.txt', spe_file= 'SPE_ChEMBL.txt')

from datasets import load_dataset
# If the dataset is gated/private, make sure you have run huggingface-cli login
raw_dataset = load_dataset("phanvancongthanh/enamine_diversity", num_proc=96,token = DATA_TOKEN_HF)
raw_dataset["train"] = load_dataset("phanvancongthanh/enamine_diversity", split="train[5%:]")
raw_dataset["validation"] = load_dataset("phanvancongthanh/enamine_diversity", split="train[:5%]")

def tokenize_function(examples):
    return tokenizer(examples["SMILES"], return_special_tokens_mask=True)

tokenized_datasets = raw_dataset.map(tokenize_function, batched=True, num_proc=96, remove_columns=raw_dataset["train"].column_names)

# For trainning 64 model
max_seq_length= 64

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // max_seq_length) * max_seq_length
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result

# Group the text for efficent trainning
tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=96)

# Configuration for models
language = "smiles"
model_config = "roberta-base"
model_dir = model_config + f"-pretrained-{language}"

Path(model_dir).mkdir(parents=True, exist_ok=True)

config = AutoConfig.from_pretrained(model_config)

config = RobertaConfig(
    classifier_dropout = 0.1, # Change if overfit but be awared of underfit
    vocab_size = 3132, # Vocab size of the tokenizer
    max_position_embeddings=66, # Train 64 max length
)

config.save_pretrained(f"{model_dir}")

# Setting for training parameters
per_device_batch_size = 512
num_epochs = 40
training_seed = 0
learning_rate = 5e-5

#setup warmup steps
warmup_ratio = 0.1  # 10% of total training steps
steps_per_epoch = len(tokenized_datasets['train']) // per_device_batch_size
total_steps = num_epochs * steps_per_epoch  # Calculate total steps based on your dataset and batch size
warmup_steps = int(warmup_ratio * total_steps)  # Calculate warmup steps


total_batch_size = per_device_batch_size * jax.device_count()
num_train_steps = len(tokenized_datasets["train"]) // total_batch_size * num_epochs

from transformers import FlaxAutoModelForMaskedLM

model = FlaxAutoModelForMaskedLM.from_config(config,
                                             seed=training_seed,
                                             dtype=jnp.dtype("bfloat16"))

# Set up warm up steps
warmup_fn = optax.linear_schedule(
    init_value=0.0, end_value=learning_rate, transition_steps=warmup_steps
)

decay_fn = optax.linear_schedule(
    init_value=learning_rate,
    end_value=0,
    transition_steps=num_train_steps - warmup_steps,
)

linear_decay_lr_schedule_fn = optax.join_schedules(
    schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps]
)

adamw = optax.adamw(learning_rate=linear_decay_lr_schedule_fn,
                    b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01)

optimizer = optax.MultiSteps(
    adamw, 4
)

state = train_state.TrainState.create(apply_fn=model.__call__,
                                      params=model.params,
                                      tx=optimizer)

@flax.struct.dataclass
class FlaxDataCollatorForMaskedLanguageModeling:
    mlm_probability: float = 0.15

    def __call__(self, examples, tokenizer, pad_to_multiple_of=16):
        batch = tokenizer.pad(examples, return_tensors="np", pad_to_multiple_of=pad_to_multiple_of)

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"], special_tokens_mask, tokenizer
        )

        return batch

    def mask_tokens(self, inputs, special_tokens_mask, tokenizer):
        labels = inputs.copy()
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        special_tokens_mask = special_tokens_mask.astype("bool")

        probability_matrix[special_tokens_mask] = 0.0
        masked_indices = np.random.binomial(1, probability_matrix).astype("bool")
        labels[~masked_indices] = -100 

        indices_replaced = np.random.binomial(1, np.full(labels.shape, 0.8)).astype("bool") & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        indices_random = np.random.binomial(1, np.full(labels.shape, 0.5)).astype("bool")
        indices_random &= masked_indices & ~indices_replaced
        random_words = np.random.randint(tokenizer.vocab_size, size=labels.shape, dtype="i4")
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

data_collator = FlaxDataCollatorForMaskedLanguageModeling(mlm_probability=0.15)

def generate_batch_splits(num_samples, batch_size, rng=None):
    samples_idx = jax.numpy.arange(num_samples)

    if input_rng is not None:
        samples_idx = jax.random.permutation(input_rng, samples_idx)

    samples_to_remove = num_samples % batch_size

    if samples_to_remove != 0:
        samples_idx = samples_idx[:-samples_to_remove]

    batch_idx = np.split(samples_idx, num_samples // batch_size)
    return batch_idx

def train_step(state, batch, dropout_rng):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        labels = batch.pop("labels")
        logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
        label_mask = jax.numpy.where(labels > 0, 1.0, 0.0)
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])) * label_mask
        loss = loss.sum() / label_mask.sum()

        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)

    metrics = jax.lax.pmean(
        {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
    )

    return new_state, metrics, new_dropout_rng

def eval_step(params, batch):
    labels = batch.pop("labels")
    logits = model(**batch, params=params, train=False)[0]
    label_mask = jax.numpy.where(labels > 0, 1.0, 0.0)
    loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])) * label_mask
    accuracy = jax.numpy.equal(jax.numpy.argmax(logits, axis=-1), labels) * label_mask
    metrics = {"loss": loss.sum(), "accuracy": accuracy.sum(), "normalizer": label_mask.sum()}
    metrics = jax.lax.psum(metrics, axis_name="batch")
    return metrics

def process_eval_metrics(metrics):
    metrics = get_metrics(metrics)
    metrics = jax.tree_map(jax.numpy.sum, metrics)
    normalizer = metrics.pop("normalizer")
    metrics = jax.tree_map(lambda x: x / normalizer, metrics)
    return metrics

parallel_train_step = jax.pmap(train_step, "batch")
parallel_eval_step = jax.pmap(eval_step, "batch")
state = flax.jax_utils.replicate(state)
rng = jax.random.PRNGKey(training_seed)
dropout_rngs = jax.random.split(rng, jax.local_device_count())

wandb.login(key=YOUR_TOKEN_WANDB)
wandb.init(project=YOUR_PROJECT_NAME)
config = {
    'learning_rate': learning_rate,
    'batch_size': per_device_batch_size,
    'num_epochs': num_epochs,
}
wandb.config.update(config)

output_dir = 'SMILES-40epochs-model'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

repo_name =  Path(output_dir).absolute().name
repo_id = create_repo(repo_name, exist_ok=True, token=YOUR_TOKEN_HF).repo_id
repo = Repository(output_dir, clone_from=repo_id, token=YOUR_TOKEN_HF)

for epoch in tqdm(range(1, num_epochs + 1), desc=f"Epoch ...", position=0, leave=True):
    rng, input_rng = jax.random.split(rng)
    train_batch_idx = generate_batch_splits(len(tokenized_datasets["train"]), total_batch_size, rng=input_rng)

    with tqdm(total=len(train_batch_idx), desc="Training...", leave=False) as progress_bar_train:
        for batch_idx in train_batch_idx:
            model_inputs = data_collator(tokenized_datasets["train"][batch_idx], tokenizer=tokenizer, pad_to_multiple_of=16)
            model_inputs = shard(model_inputs.data)
            state, train_metric, dropout_rngs = parallel_train_step(state, model_inputs, dropout_rngs)

            progress_bar_train.update(1)

        progress_bar_train.write(
              f"Train... ({epoch}/{num_epochs} | Loss: {round(train_metric['loss'].mean(), 3)}, Learning Rate: {round(train_metric['learning_rate'].mean(), 6)})"
        )

    eval_batch_idx = generate_batch_splits(len(tokenized_datasets["validation"]), total_batch_size)
    eval_metrics = []

    with tqdm(total=len(eval_batch_idx), desc="Evaluation...", leave=False) as progress_bar_eval:
        for batch_idx in eval_batch_idx:
            model_inputs = data_collator(tokenized_datasets["validation"][batch_idx], tokenizer=tokenizer)
            model_inputs = shard(model_inputs.data)
            eval_metric = parallel_eval_step(state.params, model_inputs)
            eval_metrics.append(eval_metric)

            progress_bar_eval.update(1)

        eval_metrics_dict = process_eval_metrics(eval_metrics)
        progress_bar_eval.write(
            f"Eval... ({epoch}/{num_epochs} | Loss: {eval_metrics_dict['loss']}, Acc: {eval_metrics_dict['accuracy']})"
        )

    wandb.log({
        'train_loss': float(np.asarray(train_metric['loss'].mean())),
        'learning_rate': float(np.asarray(train_metric['learning_rate'].mean()))
    }, step=epoch)

    wandb.log({
        'eval_loss': float(np.asarray(eval_metrics_dict['loss'])),
        'accuracy': float(np.asarray(eval_metrics_dict['accuracy']))
    }, step=epoch)

    epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(epoch_output_dir, exist_ok=True)

    params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
    model.save_pretrained(epoch_output_dir, params=params)

    wandb.save(epoch_output_dir + '/*')

    repo.push_to_hub(commit_message=f"Saving weights and logs of epoch {epoch}", blocking=False)
