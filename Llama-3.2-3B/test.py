import random

digits = []
for i in range(3):
    digit = random.randint(0, 99)
    digits.append(digit)
    

for digit in digits:
    print(digit, end=' ')
print()

#output: 40 97 0