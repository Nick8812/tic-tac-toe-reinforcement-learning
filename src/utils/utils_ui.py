
# This script implements some utilities for the training analysis

import numpy as np

def toSymbolValues(x):
    """
    Auxiliary method for obsToBoard
        
    OUTPUT: int
    """
    if x == 1:
        return -1
    elif x == 2:
        return 1
    else:
        return 0

def obsToBoard(n):
    """
    It converts the observation value into board values
    
    INPUT: int, n is the observation value
    OUTPUT: np.array, board integer representation 
    
    EXAMPLE: obsToBoard(745) = array([[-1, 1, -1],[0, 0, 0],[-1, 0, 0]])
    """
    if n == 0:
        return '0'
    
    nums = []
    
    while n:
        n, r = divmod(n, 3)
        nums.append(r)
      
    nums = list(map(toSymbolValues, nums))
    nums = nums + list(np.zeros([9-len(nums)], dtype = np.int))
    nums = np.array(nums).reshape(3,3)
    
    return nums

def renderQOnBoard(env, obs, Q):
    '''
    Visualizing the current board situation with Q-values printed on it
    '''
    board = obsToBoard(obs)
    
    for i in range(3):
        space = '--------------------'
        print(space)
        for j in range(3):
            print("  ", end = "")
            if board[i, j] == env.x:
                print("x ", end = "")
            elif board[i, j] == env.o:
                print("o ", end = "")
            else:
                idx = {value: key for key, value in env.dict_idx_actions.items()}[(i,j)]
                q = Q[obs, idx]
                value = 0 if q == 0. else round(q,3)
                print(str(value) + ' ', end = "")
        print("")
    print(space)