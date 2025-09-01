from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os

def check_datasets():
    """Download and check a small subset of translation data for learning"""
    print("🔍 Checking available datasets...")
    
    try:
        # Load a small subset of the WMT19 dataset for English-Chinese translation
        dataset = load_dataset("wmt19", "zh-en", split="train")
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Total samples: {len(dataset)}")
        
        # Take only a small subset for learning (first 1000 samples)
        small_dataset = dataset.select(range(min(1000, len(dataset))))
        print(f"📚 Using small subset: {len(small_dataset)} samples")
        
        # Show some examples
        print("\n📝 Sample data:")
        for i in range(min(3, len(small_dataset))):
            print(f"Sample {i+1}:")
            print(f"  English: {small_dataset[i]['translation']['en']}")
            print(f"  Chinese: {small_dataset[i]['translation']['zh']}")
            print()
        
        # Save a small subset to local files for easy access
        save_small_dataset(small_dataset, max_samples=100)
        
        return small_dataset
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("🔄 Creating synthetic data for testing...")
        return create_synthetic_data()

def save_small_dataset(dataset, max_samples=100):
    """Save a small subset of the dataset to local files"""
    print(f"💾 Saving {max_samples} samples to local files...")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Save training data
    with open("data/train.en-zh.txt", "w", encoding="utf-8") as f:
        for i in range(min(max_samples, len(dataset))):
            en_text = dataset[i]['translation']['en']
            zh_text = dataset[i]['translation']['zh']
            f.write(f"{en_text}\t{zh_text}\n")
    
    # Save validation data (next 20 samples)
    with open("data/val.en-zh.txt", "w", encoding="utf-8") as f:
        for i in range(max_samples, min(max_samples + 20, len(dataset))):
            en_text = dataset[i]['translation']['en']
            zh_text = dataset[i]['translation']['zh']
            f.write(f"{en_text}\t{zh_text}\n")
    
    print("✅ Data saved to data/train.en-zh.txt and data/val.en-zh.txt")

def create_synthetic_data():
    """Create synthetic English-Chinese pairs for testing when dataset is unavailable"""
    print("🔧 Creating synthetic translation data...")
    
    synthetic_data = [
        ("Hello world", "你好世界"),
        ("How are you", "你好吗"),
        ("I love you", "我爱你"),
        ("Good morning", "早上好"),
        ("Thank you", "谢谢"),
        ("Goodbye", "再见"),
        ("What is your name", "你叫什么名字"),
        ("I am learning", "我在学习"),
        ("Machine learning", "机器学习"),
        ("Artificial intelligence", "人工智能"),
        ("Deep learning", "深度学习"),
        ("Neural network", "神经网络"),
        ("Transformer model", "Transformer模型"),
        ("Natural language processing", "自然语言处理"),
        ("Computer science", "计算机科学"),
        ("Programming", "编程"),
        ("Python language", "Python语言"),
        ("Data science", "数据科学"),
        ("Big data", "大数据"),
        ("Cloud computing", "云计算")
    ]
    
    # Save synthetic data
    os.makedirs("data", exist_ok=True)
    
    with open("data/train.en-zh.txt", "w", encoding="utf-8") as f:
        for en, zh in synthetic_data[:15]:  # First 15 for training
            f.write(f"{en}\t{zh}\n")
    
    with open("data/val.en-zh.txt", "w", encoding="utf-8") as f:
        for en, zh in synthetic_data[15:]:  # Last 5 for validation
            f.write(f"{en}\t{zh}\n")
    
    print("✅ Synthetic data created!")
    print(f"📚 Training samples: 15")
    print(f"📚 Validation samples: 5")
    
    return synthetic_data

def test_data_loading():
    """Test if the saved data can be loaded correctly"""
    print("\n🧪 Testing data loading...")
    
    try:
        # Load training data
        with open("data/train.en-zh.txt", "r", encoding="utf-8") as f:
            train_lines = f.readlines()
        
        # Load validation data
        with open("data/val.en-zh.txt", "r", encoding="utf-8") as f:
            val_lines = f.readlines()
        
        print(f"✅ Training data: {len(train_lines)} samples")
        print(f"✅ Validation data: {len(val_lines)} samples")
        
        # Show first few samples
        print("\n📝 First training samples:")
        for i, line in enumerate(train_lines[:3]):
            en, zh = line.strip().split('\t')
            print(f"  {i+1}. English: {en}")
            print(f"     Chinese: {zh}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading saved data: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting dataset preparation for Transformer learning...")
    print("=" * 60)
    
    # Download/check dataset
    dataset = check_datasets()
    
    # Test data loading
    test_data_loading()
    
    print("\n🎯 Next steps:")
    print("1. Your data is ready in the 'data/' folder")
    print("2. You can now run the Transformer training")
    print("3. Start with a small model configuration for learning")
    print("\n💡 Learning path:")
    print("   - First: Understand the data format")
    print("   - Second: Test the Transformer forward pass")
    print("   - Third: Train on the small dataset")
    print("   - Fourth: Experiment with different configurations")
