# crf_utils.py
def prepare_crf_data(tokens, entropy_scores, threshold=0.7):
    """
    Prepares data in a format suitable for CRF training.

    Args:
        tokens (list): List of tokens (words).
        entropy_scores (list): List of entropy scores corresponding to each token.
        threshold (float): Threshold to determine 'B' (boundary) or 'O' (other).

    Returns:
        tuple: A tuple containing:
            - X_seq (list of list of dict): Features for CRF.
            - y_seq (list of list of str): Labels for CRF.
    """
    X_seq = []
    y_seq = []

    # Features for the current token
    current_token_features = []
    for i, token in enumerate(tokens):
        features = {
            'token': token,
            'entropy': entropy_scores[i],
            'capitalized': token[0].isupper(),
            'ends_with_dot': token.endswith('.'),
            # Add more features as needed for CRF
            'is_punctuation': token in ['.', '!', '?'],
            'is_numeric': token.replace('.', '', 1).isdigit(), # checks if token is numeric (allows one decimal point)
            'token_length': len(token),
        }
        # Add contextual features (previous/next tokens' properties)
        if i > 0:
            features['prev_token'] = tokens[i-1]
            features['prev_entropy'] = entropy_scores[i-1]
            features['prev_capitalized'] = tokens[i-1][0].isupper()
        else:
            features['prev_token'] = 'BOS' # Beginning of Sequence
            features['prev_entropy'] = 0.0
            features['prev_capitalized'] = False

        if i < len(tokens) - 1:
            features['next_token'] = tokens[i+1]
            features['next_entropy'] = entropy_scores[i+1]
            features['next_capitalized'] = tokens[i+1][0].isupper()
        else:
            features['next_token'] = 'EOS' # End of Sequence
            features['next_entropy'] = 0.0
            features['next_capitalized'] = False

        current_token_features.append(features)

    X_seq.append(current_token_features)

    # Labels for the current token
    current_token_labels = []
    for i, entropy in enumerate(entropy_scores):
        label = 'B' if entropy > threshold else 'O'
        current_token_labels.append(label)
    y_seq.append(current_token_labels)

    return X_seq, y_seq
