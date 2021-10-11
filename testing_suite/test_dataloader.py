import caracnl
import numpy as np


class TestingSuite:
    def test_read_file_without_amp(self):
        directory = ("C:\\Users\\utric\\Documents\\BioMaps\\4. Analyse VNA\\DATA\\"
                     "2018-III-Temperature_vs_puissance-80K\\Span=1MHz")
        omega, S11, P_VNA = caracnl.dataloader.get_s11_data_from_file(directory)

        caracnl.display_s11(P_VNA,omega, S11)


if __name__ == "__main__":
    testing_suite = TestingSuite()
    testing_suite.test_read_file_without_amp()

