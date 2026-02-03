# 1. Install Git LFS
sudo apt-get install git-lfs
git lfs install

# 2. Clone Med-Flamingo
git clone https://github.com/snap-stanford/med-flamingo.git
cd med-flamingo
source install.sh

# 3. Download Llama-7B (Med-Flamingo's language backbone)
mkdir models && cd models
git clone https://huggingface.co/decapoda-research/llama-7b-hf

# 4. Fix tokenizer config (CRITICAL)
# Edit llama-7b-hf/tokenizer_config.json
# Change: "tokenizer_class": "LlamaTokenizer"

# 5. Download Med-Flamingo checkpoint
huggingface-cli download med-flamingo/med-flamingo --local-dir ./checkpoints/med-flamingo