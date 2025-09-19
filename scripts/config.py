"""
Centralized configuration for the Dual LSTM Bitcoin prediction model.
Change parameters here and they will be automatically used across all scripts.
"""

# Model Architecture Configuration
MODEL_CONFIG = {
    'use_attention': True,
    'use_cross_attention': True,
    'use_positional_encoding': True,
    'use_layer_norm': True,
    'use_input_projection': True,
    # Architecture parameters
    'num_attention_heads': 6,              
    'regression_hidden_size': 120,          
    'positional_encoding_max_len': 60,     
    'log_constant': 10000.0
}

# Training Configuration
TRAINING_CONFIG = {
    'num_epochs': 100,                    
    'batch_size': 32,                     
    'learning_rate': 0.0015,               
    'hidden_size': 180,                   
    'num_layers': 3,                      
    'dropout': 0.2,                      
    'sequence_length': 15,                
    'early_stopping_patience': 25,       
    'early_stopping_min_delta': 0.0001,  
    'lr_scheduler_patience': 6,           
    'lr_scheduler_factor': 0.7,           
    'gradient_clip_max_norm': 0.8         
}

# Data Configuration
DATA_CONFIG = {
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'random_state': 42,
    
    'log_epsilon': 1e-8,
    'variance_threshold': 0.001,
    'slope_threshold': 0.01
}

def get_model_config():
    """Get model architecture configuration"""
    return MODEL_CONFIG.copy()

def get_training_config():
    """Get training configuration"""
    return TRAINING_CONFIG.copy()

def get_data_config():
    """Get data configuration"""
    return DATA_CONFIG.copy()

def print_config():
    """Print current configuration"""
    print("="*50)
    print("CURRENT CONFIGURATION")
    print("="*50)
    print("Model Architecture:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nTraining:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nData Splits:")
    for key, value in DATA_CONFIG.items():
        print(f"  {key}: {value}")
    print("="*50)
