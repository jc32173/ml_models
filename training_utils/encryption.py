import pickle as pk
from cryptography.fernet import Fernet


# Pickle and encrypt data:
def encrypt_data(data, key=None, key_filename='encyption_key.txt'):
    """
    Function to pickle and encrypt data.
    """
    if key is None:
        key = Fernet.generate_key()
    key_file = open(key_filename, 'w')
    #key_file.write(trained_model_file + ': ' + key.decode("utf-8"))
    key_file.write(key.decode("utf-8"))
    key_file.close()
    f = Fernet(key)
    data_pk = pk.dumps(data)
    data = f.encrypt(data_pk)
    return data


# Decrypt encrypted pickled data:
def decrypt_data(key, encrypted_data):
    """
    Function to decrypt and unpickle data.
    """
    f = Fernet(key.encode("utf-8"))
    decrypted_data = f.decrypt(encrypted_data)
    data = pk.loads(decrypted_data)
    return data
