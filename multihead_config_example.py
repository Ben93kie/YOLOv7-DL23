#!/usr/bin/env python3
"""
Example configuration for MultiHeadDetect module
"""

# Example head configurations for different use cases

# Configuration 1: Vehicle detection specialization
VEHICLE_DETECTION_CONFIG = [
    {
        'name': 'vehicles', 
        'classes': [2, 3, 5, 7, 8],  # car, motorcycle, bus, truck, boat
        'weight': 1.2
    },
    {
        'name': 'traffic_signs', 
        'classes': [9, 10, 11, 12, 13],  # traffic light, fire hydrant, stop sign, parking meter, bench
        'weight': 1.5
    },
    {
        'name': 'general', 
        'classes': list(range(14, 80)),  # all other classes
        'weight': 1.0
    }
]

# Configuration 2: Security/surveillance specialization
SECURITY_CONFIG = [
    {
        'name': 'people', 
        'classes': [0],  # person
        'weight': 2.0  # Higher weight for person detection
    },
    {
        'name': 'vehicles', 
        'classes': [2, 3, 5, 7],  # car, motorcycle, bus, truck
        'weight': 1.5
    },
    {
        'name': 'objects_of_interest', 
        'classes': [24, 26, 28],  # backpack, handbag, suitcase
        'weight': 1.8
    },
    {
        'name': 'other', 
        'classes': [i for i in range(80) if i not in [0, 2, 3, 5, 7, 24, 26, 28]],
        'weight': 0.8
    }
]

# Configuration 3: Indoor scene understanding
INDOOR_CONFIG = [
    {
        'name': 'people_and_animals', 
        'classes': [0, 16, 17, 18, 19, 20, 21, 22, 23],  # person, cat, dog, horse, sheep, cow, elephant, bear, zebra
        'weight': 1.3
    },
    {
        'name': 'furniture', 
        'classes': [56, 57, 58, 59, 60, 61, 62],  # chair, couch, potted plant, bed, dining table, toilet, tv
        'weight': 1.1
    },
    {
        'name': 'electronics', 
        'classes': [63, 64, 65, 66, 67, 68, 72, 73, 76],  # laptop, mouse, remote, keyboard, cell phone, microwave, tv, book, scissors
        'weight': 1.0
    },
    {
        'name': 'other_indoor', 
        'classes': [i for i in range(80) if i not in [0, 16, 17, 18, 19, 20, 21, 22, 23, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 72, 73, 76]],
        'weight': 0.9
    }
]

# Configuration 4: Simple binary specialization
BINARY_CONFIG = [
    {
        'name': 'high_priority', 
        'classes': [0, 2, 3, 5, 7],  # person, car, motorcycle, bus, truck
        'weight': 1.5
    },
    {
        'name': 'low_priority', 
        'classes': [i for i in range(80) if i not in [0, 2, 3, 5, 7]],
        'weight': 1.0
    }
]

def get_config_by_name(config_name):
    """Get configuration by name"""
    configs = {
        'vehicle': VEHICLE_DETECTION_CONFIG,
        'security': SECURITY_CONFIG,
        'indoor': INDOOR_CONFIG,
        'binary': BINARY_CONFIG
    }
    return configs.get(config_name, None)

def print_config_info(config, config_name):
    """Print information about a configuration"""
    print(f"\n{config_name.upper()} Configuration:")
    print(f"Number of heads: {len(config)}")
    for head in config:
        print(f"  - {head['name']}: {len(head['classes'])} classes, weight: {head['weight']}")
        print(f"    Classes: {head['classes'][:10]}{'...' if len(head['classes']) > 10 else ''}")

if __name__ == "__main__":
    # Print all configurations
    configs = [
        (VEHICLE_DETECTION_CONFIG, "Vehicle Detection"),
        (SECURITY_CONFIG, "Security/Surveillance"),
        (INDOOR_CONFIG, "Indoor Scene"),
        (BINARY_CONFIG, "Binary Priority")
    ]
    
    for config, name in configs:
        print_config_info(config, name)