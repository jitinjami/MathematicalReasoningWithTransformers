class Vocabulary:
    '''
    Vocabulary class that keeps track of the different characters 
    and assigns an id for each unique character in the input text.
    '''

    def __init__(self, pad_token="<pad>", unk_token='<unk>', eos_token='<eos>',
                 sos_token='<sos>'):
        self.id_to_string = {} #Dict mapping from id to unique string
        self.string_to_id = {} #Dict mapping from unique string to id
        
        # add the default pad token
        self.id_to_string[0] = pad_token
        self.string_to_id[pad_token] = 0
        
        # add the default unknown token
        self.id_to_string[1] = unk_token
        self.string_to_id[unk_token] = 1
        
        # add the default unknown token
        self.id_to_string[2] = eos_token
        self.string_to_id[eos_token] = 2   

        # add the default unknown token
        self.id_to_string[3] = sos_token
        self.string_to_id[sos_token] = 3

        # shortcut access
        self.pad_id = 0
        self.unk_id = 1
        self.eos_id = 2
        self.sos_id = 3

    def __len__(self):
        '''
        Returns size of the vocabulary dictionary
        '''
        return len(self.id_to_string)

    def add_new_word(self, string):
        '''
        Add new string to the vocab dictionary
        '''
        self.string_to_id[string] = len(self.string_to_id)
        self.id_to_string[len(self.id_to_string)] = string

    # Given a string, return ID
    # if extend_vocab is True, add the new word
    def get_idx(self, string, extend_vocab=False):
        '''
        Get the id of a string, create a new entry if string doesn't exist in vocab
        '''
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            self.add_new_word(string)
            return self.string_to_id[string]
        else:
            return self.unk_id        