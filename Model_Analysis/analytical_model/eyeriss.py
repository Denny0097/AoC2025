from dataclasses import dataclass, asdict

from layer_info import Conv2DShapeParam, MaxPool2DShapeParam
from math import ceil

# Memory
DATA_SIZE = 1  # Byte
PSUM_DATA_SIZE = 4  # Byte
BUS_BANDWIDTH = 4  # Byte

# Time
CLOCK_RATE = 200 * 1e6  # 200 MHz
TIME_UNIT = 1  # cycle
SPAD_ACCESS_TIME = 1 * TIME_UNIT
GLB_ACCESS_TIME = 2 * TIME_UNIT
DRAM_ACCESS_TIME = 5 * TIME_UNIT

# Energy
ENERGY_UNIT = 1e-6  # 1 pJ = 10^6 uJ
ENERGY_PER_MAC = 2 * ENERGY_UNIT
ENERGY_PER_GLB_ACCESS = 10 * ENERGY_UNIT
ENERGY_PER_DRAM_ACCESS = 200 * ENERGY_UNIT
POWER_UNIT = 1  # 1 uW
POWER_LEAKAGE = 50 * POWER_UNIT

######################################################################################################
# N: number of ifmaps/ofmaps
# M: number of filters
# H/W: ifmap height/width
# R/S: filter height/width
# E/F: ofmap height/width
# U: stride
#  ----------------------------------------------------------------------------------------------
# m: ofmap channels in global buffer
# n: number of ifmaps in a pass
# e: width of PE-set
# p: number of filters in a pass
# q: (ifmap or filter) channels in a pass
# r: number of PE sets for different (ifmap/filter) channels
# t: number of PE sets for different filters
#  ----------------------------------------------------------------------------------------------
#  Naming Convention
# *_per_pass: compute / storage size required per pass
# *_per_layer: compute / storage size required per layer
######################################################################################################


@dataclass
class EyerissHardwareParam:
    pe_array_h: int
    pe_array_w: int
    ifmap_spad_size: int
    filter_spad_size: int
    psum_spad_size: int
    glb_size: int
    bus_bw: int
    noc_bw: int


@dataclass
class EyerissMappingParam:
    m: int  # number of ofmap channels stored in global buffer
    n: int  # number of ofmaps/ifmaps used in a processing pass
    e: int  # width of the PE set (strip-mined if nessary)
    p: int  # number of filters processed by a PE set
    q: int  # number of ifmap/filter channels processed by a PE set
    r: int  # number of PE sets for different ifmap/filter channels
    t: int  # number of PE sets for different filters


AnalysisResult = dict[str, str | int | float]


class EyerissAnalyzer:
    cnt = 0

    def __init__(
        self,
        name: str | None = None,
        hardware_param: EyerissHardwareParam | None = None,
    ) -> None:
        self.name = name if name is not None else f"mapping_{EyerissAnalyzer.cnt}"
        self._hardware = hardware_param
        self._conv_shape = None
        self._maxpool_shape = None
        self._mapping = None
        EyerissAnalyzer.cnt += 1

    @property
    def hardware(self) -> EyerissHardwareParam:
        return self._hardware

    @hardware.setter
    def hardware(self, hardware_param: EyerissHardwareParam) -> None:
        assert isinstance(hardware_param, EyerissHardwareParam)
        self._hardware = hardware_param

    @property
    def conv_shape(self) -> Conv2DShapeParam:
        return self._conv_shape

    @conv_shape.setter
    def conv_shape(self, conv_param: Conv2DShapeParam) -> None:
        assert isinstance(conv_param, Conv2DShapeParam)
        self._conv_shape = conv_param

    @property
    def maxpool_shape(self) -> MaxPool2DShapeParam:
        return self._maxpool_shape

    @maxpool_shape.setter
    def maxpool_shape(self, maxpool_param: MaxPool2DShapeParam | None) -> None:
        assert isinstance(maxpool_param, (MaxPool2DShapeParam, type(None)))
        self._maxpool_shape = maxpool_param

    @property
    def mapping(self) -> EyerissMappingParam:
        return self._mapping

    @mapping.setter
    def mapping(self, mapping_param: EyerissMappingParam) -> None:
        self._mapping = mapping_param

    # Scratchpad Memory Usage
    def filter_used(self) -> int:
        return self.mapping.q * self.conv_shape.S * self.mapping.p

    def ifmap_used(self) -> int:
        return self.mapping.q * self.conv_shape.S

    def psum_used(self) -> int:
        return self.mapping.p

    @property
    def spad_size_legal(self) -> dict[str, bool]:
        return {
            "ifmap": self.ifmap_used() <= self.hardware.ifmap_spad_size,
            "filter": self.filter_used() <= self.hardware.filter_spad_size,
            "psum": self.psum_used() <= self.hardware.psum_spad_size,
        }

    # Global Buffer (GLB) Usage
    @property
    def glb_usage_per_pass(self) -> dict[str, int]:
        sizes: dict[str, int] = {}
        #! <<<========= Implement here =========>>>
        N = self.conv_shape.N
        C = self.conv_shape.C
        H = self.conv_shape.H
        W = self.conv_shape.W
        R = self.conv_shape.R
        S = self.conv_shape.S
        E = self.conv_shape.E
        F = self.conv_shape.F
        M = self.conv_shape.M
        U = self.conv_shape.U
        
        m_ = self.mapping.m
        n_ = self.mapping.n
        e_ = self.mapping.e
        p_ = self.mapping.p
        q_ = self.mapping.q
        r_ = self.mapping.r
        t_ = self.mapping.t

        # ifmap usage
        ifmap_bytes = n_ * q_ * r_ * (U * (e_ - 1) + R) * W * DATA_SIZE
        # filter usage
        filter_bytes = p_ * t_ * q_ * r_ * R * S * DATA_SIZE
        # bias usage
        bias_bytes = m_ * PSUM_DATA_SIZE  # 4 bytes each
        # partial sum usage
        psum_bytes = n_ * m_ * e_ * F * PSUM_DATA_SIZE

        total = ifmap_bytes + filter_bytes + bias_bytes + psum_bytes
        sizes = {
            "ifmap": ifmap_bytes,
            "filter": filter_bytes,
            "bias": bias_bytes,
            "psum": psum_bytes,
            "total": total,
        }
        return sizes

    
    @property
    def glb_size_legal(self) -> bool:
        return self.glb_usage_per_pass["total"] <= self.hardware.glb_size

    # DRAM Accesses (DRAM-GLB data movement)
    @property
    def dram_access_per_layer(self) -> dict[str, int]:
        res: dict[str, int] = {}
        #! <<<========= Implement here =========>>>

        N = self.conv_shape.N
        C = self.conv_shape.C
        H = self.conv_shape.H
        W = self.conv_shape.W
        R = self.conv_shape.R
        S = self.conv_shape.S
        E = self.conv_shape.E
        F = self.conv_shape.F
        M = self.conv_shape.M
        U = self.conv_shape.U

        m_ = self.mapping.m
        n_ = self.mapping.n
        e_ = self.mapping.e
        p_ = self.mapping.p
        q_ = self.mapping.q
        r_ = self.mapping.r
        t_ = self.mapping.t

        # 1) ifmap read
        tile_ifmap = n_ * (q_*r_) * (U*(e_-1) + R) * W * DATA_SIZE
        num_ifmap_tiles = ceil(E/e_) * ceil(N/n_) * ceil(C/(q_*r_))
        ifmap_read = ceil(M/m_) * num_ifmap_tiles * tile_ifmap

        # 2) filter read
        tile_filter = (p_*t_) * (q_*r_) * R * S * DATA_SIZE
        num_filter_tiles = ceil(M/m_) * ceil(m_/(p_*t_)) * ceil(C/(q_*r_))
        filter_read = ceil(E/e_) * ceil(N/n_) * num_filter_tiles * tile_filter

        # 3) bias read
        tile_bias = (p_*t_) * PSUM_DATA_SIZE  # 4 bytes each bias
        num_bias_tiles = ceil(M/m_)
        bias_read = ceil(N/n_) * ceil(E/e_) * num_bias_tiles * tile_bias

        # 4) ofmap write
        tile_ofmap = m_ * n_ * e_ * F * DATA_SIZE
        ofmap_write = ceil(M/m_) * ceil(E/e_) * ceil(N/n_) * tile_ofmap 

        if self.maxpool_shape:
            ofmap_write = ofmap_write // 4

        # summary
        total_read = ifmap_read + filter_read + bias_read
        total_write = ofmap_write
        total = total_read + total_write

        return {
            "ifmap_read": ifmap_read,
            "filter_read": filter_read,
            "bias_read": bias_read,
            "ofmap_write": ofmap_write,
            "read": total_read,
            "write": total_write,
            "total": total
        }

    # GLB Accesses (GLB-Spad data movement)
    @property
    def glb_access_per_layer(self) -> dict[str, int]:
        res: dict[str, int] = {}
        #! <<<========= Implement here =========>>>

        N = self.conv_shape.N
        C = self.conv_shape.C
        H = self.conv_shape.H
        W = self.conv_shape.W
        R = self.conv_shape.R
        S = self.conv_shape.S
        E = self.conv_shape.E
        F = self.conv_shape.F
        M = self.conv_shape.M
        U = self.conv_shape.U

        m_ = self.mapping.m
        n_ = self.mapping.n
        e_ = self.mapping.e
        p_ = self.mapping.p
        q_ = self.mapping.q
        r_ = self.mapping.r
        t_ = self.mapping.t


        # 1) ifmap GLB read
        tile_ifmap = n_ * (q_*r_) * (U*(e_-1) + R) * W * DATA_SIZE
        # 多了reuse BLG的次數
        num_ifmap_tiles = ceil(E/e_) * ceil(N/n_) * ceil(C/(q_*r_))
        ifmap_reads_per_layer = ceil(M/m_) * num_ifmap_tiles * ceil(m_/(p_*t_)) * tile_ifmap

        # 2) filter GLB read
        tile_filter = (p_*t_) * (q_*r_) * R * S * DATA_SIZE
        num_filter_tiles = ceil(M/m_) * ceil(m_/(p_*t_)) * ceil(C/(q_*r_))
        filter_reads_per_layer = ceil(E/e_) * ceil(N/n_) * num_filter_tiles * tile_filter

        # 3) bias GLB read
        tile_bias = (p_*t_) * PSUM_DATA_SIZE
        num_bias_tiles = ceil(M/m_)
        bias_reads_per_layer = ceil(E/e_) * ceil(N/n_) * num_bias_tiles * tile_bias

        # 4) psum GLB read & write
        tile_psum = n_ * (p_*t_) * e_ * F * PSUM_DATA_SIZE
        # 讀：上一次的 psum
        # 寫：這一次更新後的 psum
        psum_reads_per_layer = ceil(N/n_) * ceil(M/m_) * ceil(E/e_) * ceil(m_/(p_*t_)) * (ceil(C/(q_*r_)) - 1) * tile_psum
        psum_writes_per_layer = ceil(N/n_) * ceil(M/m_) * ceil(E/e_) * ceil(m_/(p_*t_)) * (ceil(C/(q_*r_))) * tile_psum
        
        if self.maxpool_shape:
            psum_writes_per_layer = psum_writes_per_layer // 4


        # summary
        glb_read = ifmap_reads_per_layer + filter_reads_per_layer + bias_reads_per_layer + psum_reads_per_layer
        glb_write = psum_writes_per_layer 
        glb_total = glb_read + glb_write


        return {
            "ifmap_read": ifmap_reads_per_layer,
            "filter_read": filter_reads_per_layer,
            "bias_read": bias_reads_per_layer,
            "psum_read": psum_reads_per_layer,
            "psum_write": psum_writes_per_layer,
            "read": glb_read,
            "write": glb_write,
            "total": glb_total
        }

    
    @property
    def latency_per_layer(self) -> int:
        ofmap_size = (
            self.conv_shape.N
            * self.conv_shape.M
            * self.conv_shape.E
            * self.conv_shape.F
        )
        ppu_latency_per_elem = 1 if self.maxpool_shape is None else 5

        return (
            self.glb_access_per_layer["total"] * GLB_ACCESS_TIME
            + self.dram_access_per_layer["total"] * DRAM_ACCESS_TIME
            + ofmap_size * ppu_latency_per_elem
        )

    @property
    def macs_per_layer(self) -> int:
        return (
            self.conv_shape.N
            * self.conv_shape.M
            * self.conv_shape.E
            * self.conv_shape.F
            * self.conv_shape.C
            * self.conv_shape.R
            * self.conv_shape.S
        )

    @property
    def energy_per_layer(self) -> dict[str, float]:
        compute_energy = self.macs_per_layer * ENERGY_PER_MAC
        memory_energy = (
            self.glb_access_per_layer["total"] * ENERGY_PER_GLB_ACCESS
            + self.dram_access_per_layer["total"] * ENERGY_PER_DRAM_ACCESS
        )
        leakage_energy = POWER_LEAKAGE * self.latency_per_layer / CLOCK_RATE
        total_energy = compute_energy + memory_energy + leakage_energy
        return {
            "compute": compute_energy,
            "memory": memory_energy,
            "leakage": leakage_energy,
            "total": total_energy,
        }

    @property
    def power_per_layer(self) -> dict[str, float]:
        compute_power = (
            self.energy_per_layer["compute"] / self.latency_per_layer * CLOCK_RATE
        )
        memory_power = (
            self.energy_per_layer["memory"] / self.latency_per_layer * CLOCK_RATE
        )
        leakage_power = POWER_LEAKAGE
        total_power = compute_power + memory_power + leakage_power
        return {
            "compute": compute_power,
            "memory": memory_power,
            "leakage": leakage_power,
            "total": total_power,
        }

    @property
    def operational_intensity(self) -> float:
        return self.macs_per_layer / self.dram_access_per_layer["total"]

    @property
    def peak_performance(self) -> float:
        return self.hardware.pe_array_h * self.hardware.pe_array_w  # MACs per cycle

    @property
    def peak_bandwidth(self) -> float:
        return self.hardware.bus_bw  # bytes per cycle

    @property
    def bound_by(self) -> str:
        machine_blance_point = self.peak_performance / self.peak_bandwidth
        if self.operational_intensity > machine_blance_point:
            return "compute"
        elif self.operational_intensity < machine_blance_point:
            return "memory"
        else:
            return "balanced"

    @property
    def is_compute_bound(self) -> bool:
        return self.bound_by == "compute"

    @property
    def is_memory_bound(self) -> bool:
        return self.bound_by == "memory"

    @property
    def is_balanced(self) -> bool:
        return self.bound_by == "balanced"

    @property
    def summary(self) -> AnalysisResult:
        return {
            "layer": self.name,
            **asdict(self.hardware),
            **asdict(self.mapping),
            "glb_usage": self.glb_usage_per_pass["total"],  # bytes
            "glb_access": self.glb_access_per_layer["total"],  # bytes
            "dram_access": self.dram_access_per_layer["total"],  # bytes
            "macs": self.macs_per_layer,
            "latency": self.latency_per_layer,  # cycles
                        # or any other metrics you want to include in the report
        }
