import unittest
import numpy as np
import raocp.core.constraints.cones as core_cones


class TestCones(unittest.TestCase):
    __real = core_cones.Real()
    __zero = core_cones.Zero()
    __nonnegative_orthant = core_cones.NonnegativeOrthant()
    __second_order_cone = core_cones.SecondOrderCone()
    __cartesian = core_cones.Cartesian([__real, __zero, __nonnegative_orthant, __second_order_cone])
    __num_samples = 100
    __sample_multiplier = 10
    __cone_dimension = 20
    __num_test_repeats = 100

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_dimension_check(self):
        # cone size equals vector size
        _ = core_cones._check_dimension("Real", 5, np.ones(5))

    def test_dimension_check_failure(self):
        # cone size does not equal vector size
        with self.assertRaises(ValueError):
            _ = core_cones._check_dimension("Real", 5, np.ones(6))

    def test_real_project(self):
        # create cone
        cone_type = "Real"
        real = TestCones.__real

        # create point for projection
        vector = np.array(TestCones.__sample_multiplier * np.random.rand(TestCones.__cone_dimension))\
            .reshape((TestCones.__cone_dimension, 1))

        # create points for test
        samples = [None] * TestCones.__num_samples
        for i in range(TestCones.__num_samples):
            samples[i] = np.random.randint(-100, 100, 20)  # real samples

        # test real cone
        self.assertEqual(cone_type, type(real).__name__)
        projection = real.project(vector)
        for i in range(TestCones.__num_samples):
            self.assertTrue(np.inner(vector.reshape((TestCones.__cone_dimension,))
                                     - projection.reshape((TestCones.__cone_dimension,)),
                                     samples[i].reshape((TestCones.__cone_dimension,))
                                     - projection.reshape((TestCones.__cone_dimension,))) <= 0)

    def test_real_project_dual(self):
        # create cone
        cone_type = "Real"
        real = TestCones.__real

        # create point for projection
        vector = np.array(TestCones.__sample_multiplier * np.random.rand(TestCones.__cone_dimension)) \
            .reshape((TestCones.__cone_dimension, 1))

        # create points for test
        dual_samples = [None] * TestCones.__num_samples
        for i in range(TestCones.__num_samples):
            dual_samples[i] = np.zeros(TestCones.__cone_dimension)  # real dual samples (zero)

        # test real cone
        self.assertEqual(cone_type, type(real).__name__)
        projection_onto_dual = real.project_onto_dual(vector)
        for i in range(TestCones.__num_samples):
            self.assertTrue(np.inner(vector.reshape((TestCones.__cone_dimension,))
                                     - projection_onto_dual.reshape((TestCones.__cone_dimension,)),
                                     dual_samples[i].reshape((TestCones.__cone_dimension,))
                                     - projection_onto_dual.reshape((TestCones.__cone_dimension,))) <= 0)

    def test_zero_project(self):
        # create cone
        cone_type = "Zero"
        zero = TestCones.__zero

        # create points for projection
        vector = np.array(TestCones.__sample_multiplier * np.random.rand(TestCones.__cone_dimension))\
            .reshape((TestCones.__cone_dimension, 1))
        samples = [None] * TestCones.__num_samples
        for i in range(TestCones.__num_samples):
            samples[i] = np.zeros(TestCones.__cone_dimension)  # zero samples

        # test zero
        self.assertEqual(cone_type, type(zero).__name__)
        projection = zero.project(vector)
        for i in range(TestCones.__num_samples):
            self.assertTrue(np.inner(vector.reshape((TestCones.__cone_dimension,))
                                     - projection.reshape((TestCones.__cone_dimension,)),
                                     samples[i].reshape((TestCones.__cone_dimension,))
                                     - projection.reshape((TestCones.__cone_dimension,))) <= 0)

    def test_zero_project_dual(self):
        # create cone
        cone_type = "Zero"
        zero = TestCones.__zero

        # create points for projection
        vector = np.array(TestCones.__sample_multiplier * np.random.rand(TestCones.__cone_dimension))\
            .reshape((TestCones.__cone_dimension, 1))
        dual_samples = [None] * TestCones.__num_samples
        for i in range(TestCones.__num_samples):
            dual_samples[i] = np.random.randint(-100, 100, TestCones.__cone_dimension)  # zero dual samples (real)

        # test zero dual
        self.assertEqual(cone_type, type(zero).__name__)
        projection_onto_dual = zero.project_onto_dual(vector)
        for i in range(TestCones.__num_samples):
            self.assertTrue(np.inner(vector.reshape((TestCones.__cone_dimension,))
                                     - projection_onto_dual.reshape((TestCones.__cone_dimension,)),
                                     dual_samples[i].reshape((TestCones.__cone_dimension,))
                                     - projection_onto_dual.reshape((TestCones.__cone_dimension,))) <= 0)

    def test_nonnegative_orthant_project(self):
        # create cone
        cone_type = "NonnegativeOrthant"
        nonnegative_orthant = TestCones.__nonnegative_orthant

        # create points for projection
        vector = np.array(TestCones.__sample_multiplier * np.random.rand(TestCones.__cone_dimension))\
            .reshape((TestCones.__cone_dimension, 1))
        samples = [None] * TestCones.__num_samples
        for i in range(TestCones.__num_samples):
            samples[i] = np.random.randint(0, 100, TestCones.__cone_dimension)  # non samples

        # test non
        self.assertEqual(cone_type, type(nonnegative_orthant).__name__)
        projection = nonnegative_orthant.project(vector)
        for i in range(TestCones.__num_samples):
            self.assertTrue(np.inner(vector.reshape((TestCones.__cone_dimension,))
                                     - projection.reshape((TestCones.__cone_dimension,)),
                                     samples[i].reshape((TestCones.__cone_dimension,))
                                     - projection.reshape((TestCones.__cone_dimension,))) <= 0)

    def test_nonnegative_orthant_project_dual(self):
        # create cone
        cone_type = "NonnegativeOrthant"
        nonnegative_orthant = TestCones.__nonnegative_orthant

        # create points for projection
        vector = np.array(TestCones.__sample_multiplier * np.random.rand(TestCones.__cone_dimension))\
            .reshape((TestCones.__cone_dimension, 1))
        dual_samples = [None] * TestCones.__num_samples
        for i in range(TestCones.__num_samples):
            dual_samples[i] = np.random.randint(0, 100, TestCones.__cone_dimension)  # non samples

        # test non
        self.assertEqual(cone_type, type(nonnegative_orthant).__name__)
        projection_onto_dual = nonnegative_orthant.project_onto_dual(vector)
        for i in range(TestCones.__num_samples):
            self.assertTrue(np.inner(vector.reshape((TestCones.__cone_dimension,))
                                     - projection_onto_dual.reshape((TestCones.__cone_dimension,)),
                                     dual_samples[i].reshape((TestCones.__cone_dimension,))
                                     - projection_onto_dual.reshape((TestCones.__cone_dimension,))) <= 0)

    def test_second_order_cone_project(self):
        # create cone
        cone_type = "SecondOrderCone"
        second_order_cone = TestCones.__second_order_cone

        # repeat only required here because soc cone calculation is not straightforward (and cone is self dual)
        for _ in range(TestCones.__num_test_repeats):
            # create points for projection
            vector = np.array(TestCones.__sample_multiplier * np.random.rand(TestCones.__cone_dimension))\
                .reshape((TestCones.__cone_dimension, 1))
            samples = [None] * TestCones.__num_samples
            for i in range(TestCones.__num_samples):
                s = np.random.randint(-100, 100, TestCones.__cone_dimension - 1)
                t = np.linalg.norm(s)
                samples[i] = (np.hstack((s, t)))  # soc samples

            # test soc
            self.assertEqual(cone_type, type(second_order_cone).__name__)
            projection = second_order_cone.project(vector)
            for i in range(TestCones.__num_samples):
                self.assertTrue(np.inner(vector.reshape((TestCones.__cone_dimension,))
                                         - projection.reshape((TestCones.__cone_dimension,)),
                                         samples[i].reshape((TestCones.__cone_dimension,))
                                         - projection.reshape((TestCones.__cone_dimension,))) <= 0)

    def test_second_order_cone_project_dual(self):
        # create cone
        cone_type = "SecondOrderCone"
        second_order_cone = TestCones.__second_order_cone

        # create points for projection
        vector = np.array(TestCones.__sample_multiplier * np.random.rand(TestCones.__cone_dimension))\
            .reshape((TestCones.__cone_dimension, 1))
        dual_samples = [None] * TestCones.__num_samples
        for i in range(TestCones.__num_samples):
            s = np.random.randint(-100, 100, TestCones.__cone_dimension - 1)
            t = np.linalg.norm(s)
            dual_samples[i] = (np.hstack((s, t)))  # soc samples

        # test soc
        self.assertEqual(cone_type, type(second_order_cone).__name__)
        projection_onto_dual = second_order_cone.project_onto_dual(vector)
        for i in range(TestCones.__num_samples):
            self.assertTrue(np.inner(vector.reshape((TestCones.__cone_dimension,))
                                     - projection_onto_dual.reshape((TestCones.__cone_dimension,)),
                                     dual_samples[i].reshape((TestCones.__cone_dimension,))
                                     - projection_onto_dual.reshape((TestCones.__cone_dimension,))) <= 0)

    def test_cartesian_project(self):
        # create cone
        cone_type = "Cartesian"
        cones_type = "Real x Zero x NonnegativeOrthant x SecondOrderCone"
        cartesian = TestCones.__cartesian

        # create points for projection
        num_cones = cartesian.num_cones
        vector = [None] * num_cones
        samples = []
        for i in range(num_cones):
            samples.append([None] * TestCones.__num_samples)
            vector[i] = np.array(TestCones.__sample_multiplier * np.random.rand(TestCones.__cone_dimension))\
                .reshape((TestCones.__cone_dimension, 1))

        # create set samples
        for i in range(TestCones.__num_samples):
            samples[0][i] = np.random.randint(-100, 100, TestCones.__cone_dimension)  # real samples
            samples[1][i] = np.zeros(TestCones.__cone_dimension)  # zero samples
            samples[2][i] = np.random.randint(0, 100, TestCones.__cone_dimension)  # non samples
            s = np.random.randint(-100, 100, TestCones.__cone_dimension - 1)
            t = np.linalg.norm(s)
            samples[3][i] = np.hstack((s, t))  # soc samples

        # test cartesian
        self.assertEqual(cone_type, type(cartesian).__name__)
        self.assertEqual(cones_type, cartesian.types)
        projection = cartesian.project([vector[0], vector[1], vector[2], vector[3]])
        for i in range(num_cones):
            for j in range(TestCones.__num_samples):
                self.assertTrue(np.inner((vector[i].reshape((TestCones.__cone_dimension,))
                                          - projection[i].reshape((TestCones.__cone_dimension,))),
                                         (samples[i][j].reshape((TestCones.__cone_dimension,))
                                          - projection[i].reshape((TestCones.__cone_dimension,)))) <= 0)

    def test_cartesian_project_dual(self):
        # create cone
        cone_type = "Cartesian"
        cones_type = "Real x Zero x NonnegativeOrthant x SecondOrderCone"
        cartesian = TestCones.__cartesian

        # create points for projection
        num_cones = cartesian.num_cones
        vector = [None] * num_cones
        dual_samples = []
        for i in range(num_cones):
            dual_samples.append([None] * TestCones.__num_samples)
            vector[i] = np.array(TestCones.__sample_multiplier * np.random.rand(TestCones.__cone_dimension))\
                .reshape((TestCones.__cone_dimension, 1))

        # create set samples
        for i in range(TestCones.__num_samples):
            dual_samples[0][i] = np.zeros(TestCones.__cone_dimension)  # real dual samples (zero)
            dual_samples[1][i] = np.random.randint(-100, 100, TestCones.__cone_dimension)  # zero dual samples (real)
            dual_samples[2][i] = np.random.randint(0, 100, TestCones.__cone_dimension)  # non dual samples (non)
            s = np.random.randint(-100, 100, TestCones.__cone_dimension - 1)
            t = np.linalg.norm(s)
            dual_samples[3][i] = np.hstack((s, t))  # soc samples (soc)

        # test cartesian
        self.assertEqual(cone_type, type(cartesian).__name__)
        self.assertEqual(cones_type, cartesian.types)
        projection_onto_dual = cartesian.project_onto_dual([vector[0], vector[1], vector[2], vector[3]])
        for i in range(num_cones):
            for j in range(TestCones.__num_samples):
                self.assertTrue(np.inner((vector[i].reshape((TestCones.__cone_dimension,))
                                          - projection_onto_dual[i].reshape((TestCones.__cone_dimension,))),
                                         (dual_samples[i][j].reshape((TestCones.__cone_dimension,))
                                          - projection_onto_dual[i].reshape((TestCones.__cone_dimension,)))) <= 0)


if __name__ == '__main__':
    unittest.main()

