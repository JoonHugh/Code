def caesar_decrypt(ciphertext, shift):
    result = ""
    for char in ciphertext:
        if char.isalpha():
            base = ord('A')
            result += chr((ord(char.upper()) - base - shift) % 26 + base)
        else:
            result += char
    return result

ciphertext = "IWXHFJTHIXDCXHTPHN"

for k in range(1, 26):
    decrypted = caesar_decrypt(ciphertext, k)
    print(f"k = {k}, {decrypted}")
