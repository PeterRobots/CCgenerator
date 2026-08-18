import io
import unittest
import unittest.mock
from typing import List
from utils.exceptions import (
    NotPlacedWarning,
    MissingFileError,
    InvalidFileTypeError
    )
from src.ccgenerator.__main__ import main


class TestMain(unittest.TestCase):
    '''Smoketests'''
    @unittest.mock.patch('sys.argv', ['main.py', 'tests/test_cases/test_audio.mp3'])
    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_cli(self, mock_stdout):
        '''This is a smoke test with cli'''
        # Setup
        test_commands: List[str] = []
        expected_result: str = '0,2,NORTH\n'
        # Action
        with unittest.mock.patch('whisperX.', side_effect=test_commands):
            main()
        # Assert
        self.assertEqual(mock_stdout.getvalue(), expected_result)
    #
    # @unittest.mock.patch('sys.argv', ['main.py', '--verbose'])
    # @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    # def test_verbose(self, mock_stdout):
    #     # Setup
    #     test_commands: List[str] = [
    #         'PLACE 1,1,NORTH',
    #         'END'
    #         ]
    #     expected_results: List[str] = [f'RESPONSE: Command: \'{command}\' Successful: True' for command in test_commands]
    #     # Setup
    #     # Action
    #     with unittest.mock.patch('builtins.input', side_effect=test_commands):
    #         main()
    #     for response_str, er in zip(mock_stdout.getvalue().splitlines(), expected_results):
    #         # Assert
    #         self.assertEqual(response_str, er)
    #
    # @unittest.mock.patch('sys.argv', ['main.py', '-i', 'tests/test_cases/test_case_1.txt'])
    # @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    # def test_stdio_file_input_1(self, mock_stdout):
    #     '''Test for successful use of file input test_case_1.txt.
    #     PLACE 0,0,NORTH
    #     MOVE
    #     REPORT
    #     % Output: 0,1,NORTH
    #     '''
    #     # Setup
    #     expected_result: str = '0,1,NORTH\n'
    #     # Action
    #     main()
    #     # Assert
    #     self.assertEqual(mock_stdout.getvalue(), expected_result)
    #
    # @unittest.mock.patch('sys.argv', ['main.py', '-i', 'tests/test_cases/test_case_2.txt'])
    # @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    # def test_stdio_file_input_2(self, mock_stdout):
    #     '''Test for successful use of file input test_case_2.txt.
    #     PLACE 0,0,NORTH
    #     LEFT
    #     REPORT
    #     % Output: 0,0,WEST
    #     '''
    #     # Setup
    #     expected_result: str = '0,0,WEST\n'
    #     # Action
    #     main()
    #     # Assert
    #     self.assertEqual(mock_stdout.getvalue(), expected_result)
    #
    # @unittest.mock.patch('sys.argv', ['main.py', '-i', 'tests/test_cases/test_case_3.txt'])
    # @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    # def test_stdio_file_input_3(self, mock_stdout):
    #     '''Test for successful use of file input test_case_3.txt.
    #     MOVE
    #     MOVE
    #     PLACE 1,2,EAST
    #     LEFT
    #     MOVE
    #     PLACE 1,1,EAST
    #     REPORT
    #     % Output: 1,1,EAST
    #
    #     There should be a warnings relating to not placed for first 2 commands.
    #     '''
    #     # Setup
    #     expected_result: str = '1,1,EAST\n'
    #
    #     # Assert
    #     with self.assertWarns(NotPlacedWarning):
    #         # Action
    #         main()
    #     self.assertEqual(mock_stdout.getvalue(), expected_result)
    #
    # @unittest.mock.patch('sys.argv', ['main.py', '-i', 'tests/test_cases/test_case_4.txt'])
    # @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    # def test_stdio_file_input_4(self, mock_stdout):
    #     '''Test for successful use of file input test_case_4.txt.
    #     PLACE 1,2,EAST
    #     MOVE
    #     RIGHT
    #     MOVE
    #     LEFT
    #     MOVE
    #     REPORT
    #     % Output: 3,1,EAST
    #     '''
    #     # Setup
    #     expected_result: str = '3,1,EAST\n'
    #     # Action
    #     main()
    #     # Assert
    #     self.assertEqual(mock_stdout.getvalue(), expected_result)
    #
    # @unittest.mock.patch('sys.argv', ['main.py', '-i', 'tests/test_cases/fake_test.txt'])
    # @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    # def test_stdio_missing_file_input(self, mock_stdout):
    #     '''Test for error catch with a missing file input.'''
    #     # Assert
    #     with self.assertRaises(MissingFileError):
    #         # Action
    #         main()
    #
    # @unittest.mock.patch('sys.argv', ['main.py', '-i', 'tests/test_cases/test_case_5.nottxt'])
    # @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    # def test_stdio_invalid_file_input(self, mock_stdout):
    #     '''Test for error catch with a invalid file input.'''
    #     # Assert
    #     with self.assertRaises(InvalidFileTypeError):
    #         # Action
    #         main()
