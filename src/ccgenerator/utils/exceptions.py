# EXCEPTIONS #

class BaseMissingException(Exception):
    '''Base for missing configuration.'''
    message = "default"
    def __init__(self, *args):
        super().__init__(self.message, args)

class InvalidFileTypeError(Exception):
    '''Invalid file type exception.
    Ensure file is valid and expected format.
    '''
    pass


class MissingFileError(BaseMissingException):
    '''File specified not found at path given.'''
    pass


class MissingHFToken(BaseMissingException):
    '''Missing huggingface token with write permission.'''
    message = "Missing huggingface token with write permission."
    pass


class MissingPyAnnoteToken(BaseMissingException):
    '''Missing PyAnnote token for precision model.'''
    message = "Missing PyAnnote token for precision model."
    pass

class BadModelName(BaseMissingException):
    message = "Bad model name given."

# WARNINGS #

class BaseCommandWarning(UserWarning):
    '''Base command warning, for internal use not external.'''

    def __init__(self, message, *args):
        new_message: str = f'Command ignored: {message}'
        # self.message = new_message
        super().__init__(new_message, args)


class InputWarning(BaseCommandWarning):
    '''Input warning.
        Command must be a string and one of the following:
        'PLACE X,Y,F'
            - X,Y are integers from 0 to 4
            - F is the facing direction string of 'NORTH', 'EAST', 'SOUTH' or 'WEST'
            Example: 'PLACE 0,1,NORTH'
        'MOVE'
        'LEFT'
        'RIGHT'
        'REPORT'
            - Returns the current position and facing direction.
            Example Output: 0,1,NORTH

        Ensure no extra spaces.
        A single command per line.
    '''
    pass

