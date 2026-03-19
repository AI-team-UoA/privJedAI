from openfhe import *
from typing import List


def _overlap_jaccard(cc: CryptoContext, key_pair: KeyPair, encoded_a : List[int], encoded_b : List[int]) -> float:
    plaintext_a : Plaintext = cc.MakePackedPlaintext(encoded_a)
    plaintext_b : Plaintext = cc.MakePackedPlaintext(encoded_b)
    ciphertext_b : Ciphertext = cc.Encrypt(key_pair.publicKey, plaintext_b)
    a : list = plaintext_a.GetPackedValue()
    length_b : int = plaintext_b.GetLength()
    
    if len(a) < length_b: 
        while len(a) < length_b: 
            a.append(-1)

    length_a = len(a)
            
    intersection : float = 0.0
    for _ in range(0, length_a):
        rec_a : Plaintext = cc.MakePackedPlaintext(a)
        d = cc.EvalSub(rec_a, ciphertext_b)
        decrypt_res : Plaintext = cc.Decrypt(key_pair.secretKey, d)
        decrypt_res.SetLength(length_b)
        vector_res : list = decrypt_res.GetPackedValue()
        intersection += sum(1 for v in vector_res if v == 0)
        a = a[1:] + a[:1]    
    
    if length_a + length_b - intersection == 0:
        return 0.0

    return float((intersection)/(length_a + length_b - intersection))

def _extension2_jaccard(cc: CryptoContext, key_pair: KeyPair, encoded_a : List[int], encoded_b : List[int]) -> float:
    plaintext_a : Plaintext = cc.MakePackedPlaintext(encoded_a)
    plaintext_b : Plaintext = cc.MakePackedPlaintext(encoded_b)
    length_b : int = plaintext_b.GetLength()

    ciphertext_b = list()
    for b in encoded_b:
        ciphertext_b.append(cc.Encrypt(key_pair.publicKey,
                            cc.MakePackedPlaintext([b])))

    length_a : int = plaintext_a.GetLength()
    
    cc.EvalAtIndexKeyGen(key_pair.secretKey, [-length_a])
    intersection = 0.0

    for _curr in ciphertext_b:
        curr = cc.EvalAtIndex(_curr, -length_a)
        d = cc.EvalSub(plaintext_a, cc.EvalSum(curr, length_a))
        decrypt_res : Plaintext = cc.Decrypt(key_pair.secretKey, d)
        decrypt_res.SetLength(length_a)
        vector_res : list = decrypt_res.GetPackedValue()
        intersection += sum(1 for v in vector_res if  v==0) 
        
    return float((intersection)/(length_a + length_b - intersection))
     
def _naive_jaccard(cc: CryptoContext, key_pair: KeyPair, encoded_a : List[int], encoded_b : List[int]) -> float:
    plaintext_a : list = list()
    plaintext_b : list = list()
    ciphertext_b : list = list()
    
    for a in encoded_a: 
        encrypted_a = cc.MakePackedPlaintext([a])
        plaintext_a.append(encrypted_a)
    
    for b in encoded_b: 
        encrypted_b = cc.MakePackedPlaintext([b])
        plaintext_b.append(encrypted_b)
        ciphertext_b.append(cc.Encrypt(key_pair.publicKey, encrypted_b))

    # NAIVE
    intersection = 0.0
    for plain_a in plaintext_a:
        for cipher_b in ciphertext_b: 
            sub = cc.EvalSub(plain_a, cipher_b)
            decrypt_res : Plaintext = cc.Decrypt(key_pair.secretKey, sub)
            if decrypt_res.GetPackedValue()[0] == 0 : 
                intersection += 1.0

    length_a = len(plaintext_a)
    length_b = len(ciphertext_b)
    
    return float((intersection)/(length_a + length_b - intersection))

def _extension_jaccard(cc: CryptoContext, key_pair: KeyPair, encoded_a : List[int], encoded_b : List[int]) -> float:
    plaintext_a : list = list()
    plaintext_b : list = list()
    ciphertext_b : list = list()
    
    for a in encoded_a: 
        encrypted_a = cc.MakePackedPlaintext([a])
        plaintext_a.append(encrypted_a)
    
    for b in encoded_b: 
        encrypted_b = cc.MakePackedPlaintext([b])
        plaintext_b.append(encrypted_b)
        ciphertext_b.append(cc.Encrypt(key_pair.publicKey, encrypted_b))

    # EXTENSION
    length_a = len(plaintext_a)
    length_b = len(ciphertext_b)
    

    for _ in range(length_b-1): 
        plaintext_a.extend(plaintext_a[:length_a])


    ciphertext_b_exp = list()
    for i in range(length_b):
        for _ in range(length_a): 
            ciphertext_b_exp.append(ciphertext_b[i])

    if len(plaintext_a) != len(ciphertext_b_exp):
        raise AttributeError("Expansion is wrong")
  
    intersection = 0.0
    for i in range(len(plaintext_a)):
        sub = cc.EvalSub(plaintext_a[i], ciphertext_b_exp[i])
        decrypt_res : Plaintext = cc.Decrypt(key_pair.secretKey, sub)
        if decrypt_res.GetPackedValue()[0] == 0 : 
            intersection += 1.0

    if length_a + length_b - intersection == 0:
        return 0.0
    
    return float((intersection)/(length_a + length_b - intersection))
