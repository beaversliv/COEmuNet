def load_state_dict(model, state_dict):
    """
    Load state_dict into model, handling both formats with and without 'module.' prefix.
    """
    # Create a new state dictionary to store the adjusted keys
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'module.' prefix if it exists
        if key.startswith('module.'):
            new_key = key[7:]  # Remove the first 7 characters ('module.')
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    # Load the adjusted state dictionary into the model
    model.load_state_dict(new_state_dict)