import scalesim.scale_config as config

class power_estimator:
    def __init__(self):
        self.acc_config = config

        self.leakage = []
        self.dynamic = []

        self.spsram512x128 = {}
        self.dpsram256x32 = {}

        self.freq = 0

        # IFM + Filter SRAM read
        self.avg_sm_read_bytes = 0
        # OFM SRAM write
        self.avg_sm_write_bytes = 0
        # OFM Partial Sum Buffer Read and Write
        self.avg_ps_access_bytes = 0

    def set_params(self, freq=400, cfg=config, avg_sm_read_bytes=0.0, avg_sm_write_bytes=0.0, avg_ps_access_bytes=0.0):
        self.leakage = []
        self.dynamic = []

        self.acc_config = cfg

        # SRAM power profile: read(uA/MHz) write(uA/MHz) leakage(uA)
        self.spsram512x128 = {'tt1v25c': [14.923, 15.550, 14.445], 'ff0.99v125c': [14.984, 15.526, 682.728]}
        self.dpsram256x32 = {'tt1v25c': [7.11, 5.689, 41.979], 'ff0.99v125c': [7.612, 6.006, 2121.81]}

        # clock frequency 400MHz
        self.freq = freq

        self.sram_area = 4406496
        self.non_sram_area = 566685

        self.ref_leakage_power_per_area = 1
        self.ref_switch_power_per_area = 1

        self.avg_sm_read_bytes = avg_sm_read_bytes
        self.avg_sm_write_bytes = avg_sm_write_bytes
        self.avg_ps_access_bytes = avg_ps_access_bytes

    def cal_static_power(self):
        # Memory leakage power

        # Shared Memory
        self.leakage.append(128 * self.spsram512x128['tt1v25c'][2])
        # Partial-sum buffer
        self.leakage.append(64 * self.dpsram256x32['tt1v25c'][2])

        # Other digial logic
        # self.leakage.append(self.ref_leakage_power_per_area * self.non_sram_area)

    def cal_dynamic_power(self):
        avg_sm_read_access = self.avg_sm_read_bytes / (128 / 8)
        avg_sm_write_access = self.avg_sm_write_bytes / (128 / 8)
        avg_ps_write_access = self.avg_ps_access_bytes / (32 / 8)

        sm_read_dynamic_power = (avg_sm_read_access * self.spsram512x128['tt1v25c'][0]) * self.freq
        sm_write_dynamic_power = (avg_sm_write_access * self.spsram512x128['tt1v25c'][1]) * self.freq
        ps_dynamic_power = (avg_ps_write_access * (self.dpsram256x32['tt1v25c'][0] + self.dpsram256x32['tt1v25c'][1])) * self.freq

        sm_dynamic_power = sm_read_dynamic_power + sm_write_dynamic_power

        self.dynamic.append(sm_dynamic_power)
        self.dynamic.append(ps_dynamic_power)

    def get_power(self):
        self.cal_dynamic_power()
        self.cal_static_power()

        return sum(self.dynamic) + sum(self.leakage)

if __name__ == "__main__":
    pwr = power_estimator()
    pwr.set_params(500, avg_sm_read_bytes=(64.031+15.992), avg_sm_write_bytes=16, avg_ps_access_bytes=32.093)
    print("Totoal power comsumption: (uW)")
    print(pwr.get_power())

    pwr.set_params(500, avg_sm_read_bytes=(281+16.253), avg_sm_write_bytes=1.8, avg_ps_access_bytes=3.732)
    print("Totoal power comsumption: (uW)")
    print(pwr.get_power())

    pwr.set_params(500, avg_sm_read_bytes=(15.262+30.770), avg_sm_write_bytes=15.42, avg_ps_access_bytes=30.611)
    print("Totoal power comsumption: (uW)")
    print(pwr.get_power())

    pwr.set_params(500, avg_sm_read_bytes=(87.138+33.604), avg_sm_write_bytes=1.856, avg_ps_access_bytes=3.856)
    print("Totoal power comsumption: (uW)")
    print(pwr.get_power())

