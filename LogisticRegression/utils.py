
def readfeaturelist(filename):
    """
    reads a featurelist file and returns a list of the feature names
    """
    with open(filename) as f:
        out = []        # list of feature names
        for line in f:
            out.append(line.strip())
        return out
        
def readfeatures(featurefile):
    """
    reads a featurefile consisting of userid feature;value feature;value
    returns a list where index is user id, elements are dictionaries 
    of features as keys pointing to list of values maybe should be sets
    """
    with open(featurefile) as f:
        out = [] 
        for line in f:
            tokens = line.split()
            profile = {}  # empty profile for the user
            for tok in tokens[1:]:
                feature,val = tok.rsplit(';',1)
                val = int(val)
                if feature not in profile:
                    profile[feature]=[val]
                else:
                    profile[feature].append(val)
            out.append(profile)
        for i in range(len(out)):
            assert out[i]['id'][0] == i  # check that each line was read and placed in the correct place in the list
        return out
    
def readcirclefile(circlefile):
    """
    reads a circle for a given user consisting of circleDD: user1 user2 user3 ...
    and returns a dictionary of the circle['number']=[user1,user2,user3]
    """
    with open(circlefile) as f:
        circles = {} 
        for line in f:
            tokens = line.split()
            circleID = int(tokens[0].split('circle')[1].split(':')[0])
            circles[circleID] =[]
            for tok in tokens[1:]:
              if int(tok) not in circles[circleID]:
                circles[circleID].append(int(tok))
        return circles