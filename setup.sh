#!/usr/bin/env bash
# setup.sh — إعداد البيئة الكاملة للمشروع
# شغّله مرة واحدة بعد ما تفك الـ ZIP

set -e

echo "============================================"
echo "  IMDB BERT Sentiment — Project Setup"
echo "============================================"

# 1. Check Python
python_version=$(python3 --version 2>&1)
echo "✅ Python: $python_version"

# 2. Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ venv created"
else
    echo "✅ venv already exists"
fi

# 3. Activate venv
source venv/bin/activate
echo "✅ venv activated"

# 4. Install requirements
echo ""
echo "📥 Installing requirements..."
pip install --upgrade pip -q
pip install -r requirements.txt

echo ""
echo "============================================"
echo "✅ Setup complete!"
echo ""
echo "▶️  To start Jupyter:"
echo "   source venv/bin/activate"
echo "   jupyter notebook notebooks/IMDB_BERT_Sentiment.ipynb"
echo ""
echo "▶️  To run inference after training:"
echo "   python scripts/predict.py --text \"Your review here\""
echo "============================================"
