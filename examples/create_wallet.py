# examples/create_wallet.py
from web3 import Account
import secrets

# Create a new account object using a secure source of randomness
new_account = Account.create(secrets.token_hex(32))

# Get the private key and the corresponding public address
private_key = new_account.key.hex()
public_address = new_account.address

print("--- Your New, Secure Web3 Wallet ---")
print("\nIMPORTANT: Treat the private key like a password. Do not share it.")
print("\n1. Copy the ENTIRE private key below (including the '0x').")
print(f"   Private Key: {private_key}")
print("\n2. Paste this new private key into your .env file, replacing the old one.")
print("\n3. Copy the new public address below to use in the Sepolia faucet.")
print(f"   Public Address: {public_address}")
print("\n------------------------------------")