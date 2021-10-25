import numpy as np
import caracnl


class TestingSuite:
    def test_optimize(self):
        directory = ("C:\\Users\\utric\\Documents\\BioMaps\\4. Analyse VNA\\DATA\\"
                     "2018-III-Temperature_vs_puissance-80K\\Span=1MHz")

        P_VNA, omega, S11 = caracnl.dataloader.get_s11_data_from_file(directory)
        caracnl.display_s11(P_VNA, omega, S11)

        result = caracnl.optimize.get_correction_parameters(omega, S11[0], )
        print(result)

        return result


if __name__ == "__main__":
    testing_suite = TestingSuite()

    testing_suite.test_optimize()
