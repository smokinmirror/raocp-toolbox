import unittest
import raocp.core.cones as core_cones
import numpy as np


class TestCones(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_cones_uni(self):
        # create cone
        uni = core_cones.Uni()
        cones_type = "Uni"

        # create points for projection
        num_samples = 100
        multiplier = 10
        cone_dim = 20
        x = np.array(multiplier * np.random.rand(cone_dim)).reshape((cone_dim, 1))
        samples = [None] * num_samples
        dual_samples = [None] * num_samples
        for i in range(num_samples):
            samples[i] = np.random.randint(-100, 100, 20)  # uni samples
            dual_samples[i] = np.zeros(cone_dim)  # uni dual samples (zero)

        # test uni
        self.assertEqual(cones_type, uni.type)
        projection = uni.project_onto_cone(x)
        dual_projection = uni.project_onto_dual(x)
        for i in range(len(samples)):
            self.assertTrue(np.inner(x.reshape((cone_dim,)) - projection.reshape((cone_dim,)),
                                     samples[i].reshape((cone_dim,)) - projection.reshape((cone_dim,))) <= 0)
            self.assertTrue(np.inner(x.reshape((cone_dim,)) - dual_projection.reshape((cone_dim,)),
                                     dual_samples[i].reshape((cone_dim,)) - dual_projection.reshape(
                                         (cone_dim,))) <= 0)

    def test_cones_zero(self):
        # create cone
        zero = core_cones.Zero()
        cones_type = "Zero"

        # create points for projection
        num_samples = 100
        multiplier = 10
        cone_dim = 20
        x = np.array(multiplier * np.random.rand(cone_dim)).reshape((cone_dim, 1))
        samples = [None] * num_samples
        dual_samples = [None] * num_samples
        for i in range(num_samples):
            samples[i] = np.zeros(cone_dim)  # zero samples
            dual_samples[i] = np.random.randint(-100, 100, 20)  # zero dual samples (uni)

        # test zero
        self.assertEqual(cones_type, zero.type)
        projection = zero.project_onto_cone(x)
        dual_projection = zero.project_onto_dual(x)
        for i in range(len(samples)):
            self.assertTrue(np.inner(x.reshape((cone_dim,)) - projection.reshape((cone_dim,)),
                                     samples[i].reshape((cone_dim,)) - projection.reshape((cone_dim,))) <= 0)
            self.assertTrue(np.inner(x.reshape((cone_dim,)) - dual_projection.reshape((cone_dim,)),
                                     dual_samples[i].reshape((cone_dim,)) - dual_projection.reshape(
                                         (cone_dim,))) <= 0)

    def test_cones_non(self):
        # create cone
        non = core_cones.NonnegOrth()
        cones_type = "NonnegOrth"

        # create points for projection
        num_samples = 100
        multiplier = 10
        cone_dim = 20
        x = np.array(multiplier * np.random.rand(cone_dim)).reshape((cone_dim, 1))
        samples = [None] * num_samples
        dual_samples = [None] * num_samples
        for i in range(num_samples):
            samples[i] = np.random.randint(0, 100, cone_dim)  # non samples
            dual_samples[i] = samples[i]

        # test non
        self.assertEqual(cones_type, non.type)
        projection = non.project_onto_cone(x)
        dual_projection = non.project_onto_dual(x)
        for i in range(len(samples)):
            self.assertTrue(np.inner(x.reshape((cone_dim,)) - projection.reshape((cone_dim,)),
                                     samples[i].reshape((cone_dim,)) - projection.reshape((cone_dim,))) <= 0)
            self.assertTrue(np.inner(x.reshape((cone_dim,)) - dual_projection.reshape((cone_dim,)),
                                     dual_samples[i].reshape((cone_dim,)) - dual_projection.reshape(
                                         (cone_dim,))) <= 0)

    def test_cones_soc(self):
        # create cone
        soc = core_cones.SOC()
        cones_type = "SOC"

        # create points for projection
        num_samples = 100
        multiplier = 10
        cone_dim = 20
        x = np.array(multiplier * np.random.rand(cone_dim)).reshape((cone_dim, 1))
        samples = [None] * num_samples
        dual_samples = [None] * num_samples
        for i in range(num_samples):
            s = np.random.randint(-100, 100, cone_dim - 1)
            t = np.linalg.norm(s)
            samples[i] = (np.hstack((s, t)))  # soc samples
            dual_samples[i] = samples[i]

        # test soc
        self.assertEqual(cones_type, soc.type)
        projection = soc.project_onto_cone(x)
        dual_projection = soc.project_onto_dual(x)
        for i in range(len(samples)):
            self.assertTrue(np.inner(x.reshape((cone_dim,)) - projection.reshape((cone_dim,)),
                                     samples[i].reshape((cone_dim,)) - projection.reshape((cone_dim,))) <= 0)
            self.assertTrue(np.inner(x.reshape((cone_dim,)) - dual_projection.reshape((cone_dim,)),
                                     dual_samples[i].reshape((cone_dim,)) - dual_projection.reshape(
                                         (cone_dim,))) <= 0)

    def test_cones_cart(self):
        # create cones
        uni = core_cones.Uni()
        zero = core_cones.Zero()
        non = core_cones.NonnegOrth()
        soc = core_cones.SOC()
        cones = [uni, zero, non, soc]
        cart = core_cones.Cart(cones)
        cart_type = "Uni x Zero x NonnegOrth x SOC"

        # create points for projection
        num_cones = len(cones)
        num_samples = 100
        multiplier = 10
        x = [None] * num_cones
        cone_dim = 20
        samples = []
        for i in range(num_cones * 2):
            samples.append([None] * num_samples)
        for i in range(num_cones):
            x[i] = np.array(multiplier * np.random.rand(cone_dim)).reshape((cone_dim, 1))

        # create set samples
        for i in range(num_samples):
            samples[0][i] = np.random.randint(-100, 100, 20)  # uni samples
            samples[1][i] = np.zeros(cone_dim)  # zero samples
            samples[2][i] = np.random.randint(0, 100, cone_dim)  # non samples
            s = np.random.randint(-100, 100, cone_dim - 1)
            t = np.linalg.norm(s)
            samples[3][i] = np.hstack((s, t))  # soc samples
            samples[4][i] = np.random.randint(-100, 100, cone_dim)  # uni dual samples (zero)
            samples[5][i] = np.zeros(cone_dim)  # zero dual samples (uni)
        samples[6] = samples[2]
        samples[7] = samples[3]

        # test cartesian
        self.assertEqual(cart_type, cart.type)
        projection = cart.project_onto_cone([x[0], x[1], x[2], x[3]])
        dual_projection = cart.project_onto_cone([x[0], x[1], x[2], x[3]])
        for i in range(num_cones):
            for j in range(len(samples[0])):
                self.assertTrue(np.inner((x[i].reshape((cone_dim,)) - projection[i].reshape((cone_dim,))),
                                         (samples[i][j].reshape((cone_dim,)) - projection[i].reshape(
                                             (cone_dim,)))) <= 0)
                self.assertTrue(np.inner((x[i].reshape((cone_dim,)) - dual_projection[i].reshape((cone_dim,))),
                                         (samples[i + num_cones][j].reshape((cone_dim,)) - dual_projection[
                                             i].reshape(
                                             (cone_dim,)))) <= 0)
