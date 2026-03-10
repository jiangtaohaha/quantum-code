import numpy as np
from scipy.integrate import cumulative_trapezoid
from numba import njit


# =========================================================
# Numba accelerated low-level routines
# =========================================================

@njit
def adk_array_numba(E, Ip):
    kapa = np.sqrt(2.0 * Ip)
    W = np.zeros_like(E)

    for i in range(E.size):
        Ei = abs(E[i])
        if Ei > 1e-10:
            exponent = -2.0 * kapa**3 / (3.0 * Ei)
            W[i] = (2.0 * kapa**2 / Ei)**(2.0 / kapa - 1.0) * np.exp(exponent) / 4.0
    return W


@njit
def find_crossing_idx_numba(E, dt, ti_idx, x_target):
    v = 0.0
    x = 0.0

    # 初始时刻与目标位置的相对符号
    prev_sign_pos = 0.0   # 对 +x_target
    prev_sign_neg = 0.0   # 对 -x_target

    for i in range(ti_idx, E.size):
        v += E[i] * dt
        x += v * dt

        # 相对于 +x_target 的位置
        dx_pos = x - x_target
        sign_pos = 1.0 if dx_pos > 0 else -1.0 if dx_pos < 0 else 0.0

        # 相对于 -x_target 的位置
        dx_neg = x + x_target
        sign_neg = 1.0 if dx_neg > 0 else -1.0 if dx_neg < 0 else 0.0

        if i > ti_idx:
            # 穿过 +x_target
            if sign_pos * prev_sign_pos < 0:
                return i

            # 穿过 -x_target
            if sign_neg * prev_sign_neg < 0:
                return i

        prev_sign_pos = sign_pos
        prev_sign_neg = sign_neg

    return -1


@njit
def find_second_crossing_idx_numba(E, dt, ti_idx, x_target):
    v = 0.0
    x = 0.0

    prev_sign_pos = 0.0
    prev_sign_neg = 0.0
    crossing_count = 0

    for i in range(ti_idx, E.size):
        v += E[i] * dt
        x += v * dt

        # +x_target
        dx_pos = x - x_target
        sign_pos = 1.0 if dx_pos > 0 else -1.0 if dx_pos < 0 else 0.0

        # -x_target
        dx_neg = x + x_target
        sign_neg = 1.0 if dx_neg > 0 else -1.0 if dx_neg < 0 else 0.0

        if i > ti_idx:
            if sign_pos * prev_sign_pos < 0:
                crossing_count += 1
                if crossing_count == 2:
                    return i

            if sign_neg * prev_sign_neg < 0:
                crossing_count += 1
                if crossing_count == 2:
                    return i

        prev_sign_pos = sign_pos
        prev_sign_neg = sign_neg

    return -1



# =========================================================
# Rydberg model with Numba acceleration
# =========================================================

class RydbergModelNumba:
    def __init__(self, En, omega, Ip=0.5, nt=12800):
        self.En = En
        self.omega = omega
        self.Ip = Ip

        self.T = 2.0 * np.pi / omega
        self.dur = 5.0 * self.T
        self.nt = nt

        self.ts = np.linspace(0.0, self.dur, nt)
        self.dt = self.ts[1] - self.ts[0]

    def E_field(self, t, I):
        E0 = np.sqrt(I / 351.0)
        return (
            E0
            * np.sin(np.pi * t / self.dur) ** 2
            * np.sin(self.omega * t)
        )

    def prepare(self, I, xn_target):
        # Electric field
        self.E = self.E_field(self.ts, I)

        # Vector potential
        self.A = -cumulative_trapezoid(self.E, self.ts, initial=0.0)

        # Phase integral
        phase_integrand = self.En + 0.5 * self.A**2
        self.phase_cum = cumulative_trapezoid(
            phase_integrand, self.ts, initial=0.0
        )

        # ADK rate and ground-state amplitude
        self.W = adk_array_numba(self.E, self.Ip)
        rate_int = cumulative_trapezoid(self.W, self.ts, initial=0.0)
        self.ag = np.exp(-rate_int)

        # Precompute recapture indices
        self.recapture_idx = np.empty(self.nt, dtype=np.int64)
        for ti_idx in range(self.nt):
            # self.recapture_idx[ti_idx] = find_crossing_idx_numba(
            self.recapture_idx[ti_idx] = find_second_crossing_idx_numba(
                self.E, self.dt, ti_idx, xn_target
            )

    def compute_amplitudes(self):
        amps = np.zeros(self.nt, dtype=np.complex128)
        phi_tf_all = self.phase_cum[-1]

        for ti_idx in range(self.nt):
            tm_idx = self.recapture_idx[ti_idx]
            if tm_idx >= 0 and self.W[ti_idx] > 0.0:
                phi_tm = self.phase_cum[tm_idx] - self.phase_cum[ti_idx]
                phi_tf = phi_tf_all - self.phase_cum[ti_idx]

                # amps[ti_idx] = (
                #     self.ag[ti_idx]
                #     * np.sqrt(self.W[ti_idx])
                #     * np.exp(1j * (phi_tm - phi_tf))  #  相位因子正负
                # )

                amps[ti_idx] = (
                    np.exp(1j * (phi_tm - phi_tf))  #  相位因子正负
                )
        return amps


# =========================================================
# Main test / driver
# =========================================================

def test_numba():
    nb = 6
    nI = 201
    omega = 0.056
    kapar = 0.1

    # 先创建一个临时 model，用来确定 nt
    n0 = 2
    En0 = -0.5 / n0**2
    model0 = RydbergModelNumba(En0, omega)
    nt = model0.nt

    amp_i = np.zeros((nb, nI, nt), dtype=np.complex128)
    amp_total = np.zeros((nb, nI))

    for n in range(2, nb + 2):
        xn_target = n**2
        En = -0.5 / n**2

        model = RydbergModelNumba(En, omega)

        for i in range(nI):
            I_test = 0.5 + i * 0.01
            I_coh = 1/(1+kapar) * I_test
            model.prepare(I_coh, xn_target)

            amp_i[n - 2, i, :] = model.compute_amplitudes()
            amp_total[n - 2, i] = np.trapz(
                amp_i[n - 2, i], model.ts
            )

    np.savez(
        "./classic_traj/data/ngpdwgamac2-nI201nt12800_prob.npz",
        time=model.ts,
        ampi=amp_i,
        amptotal=amp_total
    )


if __name__ == "__main__":
    test_numba()
