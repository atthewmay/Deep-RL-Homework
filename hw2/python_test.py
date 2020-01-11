def reversible(integer):
    # Eliminate leading zeros in the reversed number
    if integer % 10 == 0:
        return False
    else:
        rev_int = int(str(integer)[::-1])
        sum_int = rev_int + integer
        return all_odd(sum_int)

def all_odd(sum_int):
    for i in str(sum_int):
        if int(i) % 2 == 0:
            return False
    return True

reversible_integers = []
for i in range(1001,10002):
    if reversible(i) == True:
        reversible_integers.append(i)


print(reversible_integers)

