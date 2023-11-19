import numpy as np
import matplotlib.pyplot as plt

from current_tripole import CurrentTripole
from muscle_fiber import MuscleFiber
from motor_unit import MotorUnit
from conductor_volume import ConductorVolume

def plot_fibers(fibers):
    plt.figure()
    ax = plt.axes(projection='3d')

    for fiber in fibers:
        ax.plot([fiber.start_point[0], fiber.end_point[0]],[fiber.start_point[1], fiber.end_point[1]],[fiber.start_point[2], fiber.end_point[2]])

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    xc, yc, zc = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([xc - radius, xc + radius])
    ax.set_ylim3d([yc - radius, yc + radius])
    ax.set_zlim3d([zc - radius, zc + radius])

def plot_signals(signals, title = ''):
    n_signals = signals.shape[0]

    norm_signals = signals/np.max(np.abs(signals))
    norm_signals = norm_signals + np.arange(1, n_signals + 1).reshape((-1,1))

    plt.figure(figsize=(4, 6.5))
    ax = plt.axes()
    ax.invert_yaxis()
    ax.plot(t*1e3, norm_signals.T, 'k')
    ax.set_yticks(range(1, n_signals + 1))
    ax.set_xticks(range(0, 40, 5))
    ax.set_aspect(5)
    plt.xlim(0, 35)
    plt.grid()
    plt.title(title)
    plt.xlabel("Time (ms)")
    plt.ylabel("Normalized amplitude")

if __name__ == "__main__":
    # start_point = np.array([-1,-1,-1])
    # end_point = np.array([1,1,1]) 
    # tripole = CurrentTripole(1,1,1)
    # fibers = MuscleFiber.create_fibers(start_point, end_point, 0.6, 0.1, 0.1, 0.1, 0.5, 100, 1, tripole)
    #
    # MuscleFiber.plot_fibers(fibers)
    # plt.show()

    current_tripole = CurrentTripole(1, 7e-3*0.303, 7e-3)
    fibers = MuscleFiber.create_fibers(np.array([0, 10e-3, -90e-3 - 120e-3]),
                                       np.array([0, 10e-3, 0]),
                                       90/(90+120),
                                       10e-3, 10e-3, 5e-3, 1e-3, 100, 4,
                                       current_tripole)
    

    conductor_volume = ConductorVolume(1, 6)
    motor_unit = MotorUnit(conductor_volume, fibers)

    T = 35e-3
    sample_rate = 2000
    n_points = round(sample_rate*T)

    n_electrodes_sets = round((170-(-10))/20 + 1)

    ied = 10e-3 # inter-electrode distance
    relative_z_electrodes = ied * np.array([-1.5, -0.5, 0.5, 1.5])

    z_electrodes_sets = np.linspace(-10e-3, 170e-3, n_electrodes_sets) - 120e-3

    z_electrodes = np.reshape(relative_z_electrodes, (1,-1)) + np.reshape(z_electrodes_sets, (-1,1))
    z_electrodes = np.reshape(z_electrodes, (-1))

    xz_electrodes = np.zeros((n_electrodes_sets*4, 2))
    xz_electrodes[:, 1] = z_electrodes

    # n_repeat = 5
    # elapsed_time = timeit.timeit('motor_unit.trigger(xz_electrodes, sample_rate, n_points)', globals = globals(), number = n_repeat)
    # print('Mean elapsed time (no multiprocessing): {:.2f} s'.format(elapsed_time/n_repeat))
    # 
    # n_cores = mp.cpu_count()
    # elapsed_time = timeit.timeit('motor_unit.trigger(xz_electrodes, sample_rate, n_points, jobs=n_cores)', globals = globals(), number = n_repeat)
    # print('Mean elapsed time (multiprocessing): {:.2f} s'.format(elapsed_time/n_repeat))

    signals = motor_unit.trigger(xz_electrodes, sample_rate, n_points)

    t = np.linspace(0, (n_points-1)/sample_rate, n_points)

    VA = signals[::4,:]
    VB = signals[1::4,:]
    VC = signals[2::4,:]
    VD = signals[3::4,:]

    SD1 = VA-VB
    SD2 = VB-VC
    SD3 = VC-VD

    DD1 = SD1-SD2
    DD2 = SD2-SD3

    plot_signals(VB, 'Monopolar signals: VB')
    plot_signals(SD2[:-1,:], 'Single Diff. signals: SD2')
    plot_signals(DD1[:-2,:], 'Double Diff. signals: DD1')

    plt.show()



