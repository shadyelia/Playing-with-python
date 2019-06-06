def convert(labels, attributes):
    numericAttributes = []
    numericLabels = []
    for label in labels[1:]:
        if label == 'unacc':
            numericLabels.append(0)
        elif label == 'acc':
            numericLabels.append(1)
        elif label == 'good':
            numericLabels.append(2)
        elif label == 'vgood':
            numericLabels.append(3)

    for attribute in attributes[1:]:
        numericAttribute = []
        if(attribute[0] == 'low'):
            numericAttribute.append(0)
        elif (attribute[0] == 'med'):
            numericAttribute.append(1)
        elif(attribute[0] == 'high'):
            numericAttribute.append(2)
        elif (attribute[0] == 'vhigh'):
            numericAttribute.append(3)

        if(attribute[1] == 'low'):
            numericAttribute.append(0)
        elif (attribute[1] == 'med'):
            numericAttribute.append(1)
        elif(attribute[1] == 'high'):
            numericAttribute.append(2)
        elif (attribute[1] == 'vhigh'):
            numericAttribute.append(3)

        if(attribute[2] == 'two'):
            numericAttribute.append(0)
        elif (attribute[2] == 'three'):
            numericAttribute.append(1)
        elif(attribute[2] == 'four'):
            numericAttribute.append(2)
        elif (attribute[2] == '5more'):
            numericAttribute.append(3)

        if(attribute[3] == 'two'):
            numericAttribute.append(0)
        elif (attribute[3] == 'four'):
            numericAttribute.append(1)
        elif(attribute[3] == 'more'):
            numericAttribute.append(2)

        if(attribute[4] == 'small'):
            numericAttribute.append(0)
        elif (attribute[4] == 'med'):
            numericAttribute.append(1)
        elif(attribute[4] == 'big'):
            numericAttribute.append(2)

        if(attribute[5] == 'low'):
            numericAttribute.append(0)
        elif (attribute[5] == 'med'):
            numericAttribute.append(1)
        elif(attribute[5] == 'high'):
            numericAttribute.append(2)

        numericAttributes.append(numericAttribute)
