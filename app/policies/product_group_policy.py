from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class RoleCatalogEntry:
    role_id: str
    display_name_ru: str
    synonyms: tuple[str, ...]
    behavior: str
    quantity_rule: str
    validation_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProductGroupProfile:
    product_group_id: str
    roles: tuple[str, ...]
    required_roles: tuple[str, ...]
    optional_roles: tuple[str, ...]
    role_catalog: Mapping[str, RoleCatalogEntry] = field(default_factory=dict)
    quantity_rules: Mapping[str, str] = field(default_factory=dict)
    compatibility_dimensions: tuple[str, ...] = ()
    equivalence_rules: tuple[str, ...] = ()
    no_recommendation_rules: tuple[str, ...] = ()
    commercial_output_template: Mapping[str, object] = field(default_factory=dict)


SERVER_ROLE_CATALOG: Mapping[str, RoleCatalogEntry] = MappingProxyType(
    {
        "ready_server": RoleCatalogEntry(
            role_id="ready_server",
            display_name_ru="Готовый сервер",
            synonyms=("ready server", "server in assembly", "готовый сервер", "сервер в сборе"),
            behavior="optional",
            quantity_rule="server_qty",
            validation_capabilities=("server_base",),
        ),
        "server_platform": RoleCatalogEntry(
            role_id="server_platform",
            display_name_ru="Серверная платформа",
            synonyms=("platform", "barebone", "chassis", "серверная платформа", "шасси"),
            behavior="hard_by_default",
            quantity_rule="server_qty",
            validation_capabilities=("platform", "onboard_network"),
        ),
        "cpu": RoleCatalogEntry(
            role_id="cpu",
            display_name_ru="Процессор",
            synonyms=("cpu", "processor", "xeon", "epyc", "процессор"),
            behavior="hard_by_default",
            quantity_rule="server_qty * cpu_per_server",
            validation_capabilities=("cpu",),
        ),
        "ram": RoleCatalogEntry(
            role_id="ram",
            display_name_ru="Оперативная память",
            synonyms=("ram", "memory", "rdimm", "ddr", "память", "оперативная память"),
            behavior="hard_by_default",
            quantity_rule="server_qty * ceil(ram_gb_per_server / selected_module_gb)",
            validation_capabilities=("ram_capacity", "ram_type"),
        ),
        "storage": RoleCatalogEntry(
            role_id="storage",
            display_name_ru="Накопители",
            synonyms=("storage", "ssd", "hdd", "drive", "disk", "nvme", "накопитель", "диск"),
            behavior="hard_by_default",
            quantity_rule="server_qty * drives_per_server",
            validation_capabilities=("storage_capacity", "storage_interface"),
        ),
        "network_adapter": RoleCatalogEntry(
            role_id="network_adapter",
            display_name_ru="Сетевой адаптер",
            synonyms=("network", "nic", "ethernet", "sfp", "qsfp", "сетевой адаптер", "порт"),
            behavior="hard_when_requested",
            quantity_rule="network_adapter_ports",
            validation_capabilities=("network_ports", "network_speed", "network_media"),
        ),
        "storage_controller": RoleCatalogEntry(
            role_id="storage_controller",
            display_name_ru="Контроллер хранения",
            synonyms=("raid", "hba", "tri-mode", "controller", "fc hba", "контроллер"),
            behavior="hard_when_requested",
            quantity_rule="server_qty",
            validation_capabilities=("storage_controller", "host_adapter"),
        ),
        "gpu": RoleCatalogEntry(
            role_id="gpu",
            display_name_ru="GPU",
            synonyms=("gpu", "nvidia", "cuda", "accelerator", "graphics card", "видеокарта"),
            behavior="hard_when_requested",
            quantity_rule="server_qty * requested_gpu_per_server",
            validation_capabilities=("gpu", "accelerator"),
        ),
        "transceiver": RoleCatalogEntry(
            role_id="transceiver",
            display_name_ru="Трансивер",
            synonyms=("transceiver", "sfp module", "qsfp module", "optic", "трансивер"),
            behavior="hard_when_requested",
            quantity_rule="requested_transceivers",
            validation_capabilities=("optics_media",),
        ),
        "cable": RoleCatalogEntry(
            role_id="cable",
            display_name_ru="Кабель",
            synonyms=("cable", "dac", "aoc", "кабель"),
            behavior="hard_when_requested",
            quantity_rule="requested_cables",
            validation_capabilities=("cable_type",),
        ),
        "power_supply": RoleCatalogEntry(
            role_id="power_supply",
            display_name_ru="Блок питания",
            synonyms=("psu", "power supply", "блок питания", "бп"),
            behavior="hard_when_requested",
            quantity_rule="server_qty * requested_psu_per_server",
            validation_capabilities=("power_supply",),
        ),
        "rail_kit": RoleCatalogEntry(
            role_id="rail_kit",
            display_name_ru="Рельсы",
            synonyms=("rail", "rail kit", "rails", "рельсы"),
            behavior="hard_when_requested",
            quantity_rule="server_qty",
            validation_capabilities=("rail_kit",),
        ),
        "license": RoleCatalogEntry(
            role_id="license",
            display_name_ru="Лицензия",
            synonyms=("license", "licence", "subscription", "лицензия"),
            behavior="hard_when_requested",
            quantity_rule="requested_licenses",
            validation_capabilities=("license",),
        ),
        "support": RoleCatalogEntry(
            role_id="support",
            display_name_ru="Поддержка",
            synonyms=("support", "warranty extension", "service pack", "поддержка", "гарантия"),
            behavior="hard_when_requested",
            quantity_rule="requested_support",
            validation_capabilities=("support", "warranty"),
        ),
        "other_accessory": RoleCatalogEntry(
            role_id="other_accessory",
            display_name_ru="Прочий аксессуар",
            synonyms=("accessory", "option kit", "аксессуар", "опция"),
            behavior="optional",
            quantity_rule="requested_accessories",
            validation_capabilities=("accessory",),
        ),
    }
)


SERVER_PRODUCT_GROUP_PROFILE = ProductGroupProfile(
    product_group_id="server",
    roles=tuple(SERVER_ROLE_CATALOG),
    required_roles=("server_platform", "cpu", "ram", "storage"),
    optional_roles=tuple(
        role_id
        for role_id, role in SERVER_ROLE_CATALOG.items()
        if role.behavior != "hard_by_default"
    ),
    role_catalog=SERVER_ROLE_CATALOG,
    quantity_rules=MappingProxyType(
        {
            "server_platform": "server_qty",
            "cpu": "server_qty * cpu_per_server",
            "ram": "server_qty * ceil(ram_gb_per_server / selected_module_gb)",
            "storage": "server_qty * drives_per_server",
            "network_adapter": "server_qty * ceil(required_ports_per_server / ports_per_adapter)",
            "storage_controller": "server_qty when requested or required by selected platform",
            "gpu": "server_qty * requested_gpu_per_server",
            "transceiver": "requested_transceivers",
            "cable": "requested_cables",
            "power_supply": "server_qty * requested_psu_per_server",
            "rail_kit": "server_qty",
            "license": "requested_licenses",
            "support": "requested_support",
            "other_accessory": "requested_accessories",
        }
    ),
    compatibility_dimensions=(
        "form_factor",
        "socket",
        "cpu_family",
        "ram_type",
        "storage_interface",
        "nvme_support",
        "psu/completeness",
    ),
    equivalence_rules=(
        "repair may choose a cheaper equivalent only within the same role eligibility",
        "equivalent RAM/storage must satisfy hard capacity, interface, stock, and role rules",
        "equivalent CPU must not be worse than hard requirements and must match platform family",
    ),
    no_recommendation_rules=(
        "no complete stocked platform/CPU/RAM/storage set passes hard validation",
        "selected components have insufficient stock for calculated quantities",
        "hard compatibility dimensions conflict after validation or evidence review",
    ),
    commercial_output_template=MappingProxyType(
        {
            "default_output_mode": "single_best_cost_valid",
            "telegram": (
                "one copy-paste КП block",
                "commercial comment",
                "mandatory engineering checklist",
            ),
            "excel_sheets": ("AI-рекомендации", "Матрица компонентов"),
        }
    ),
)


NETWORK_ROLE_CATALOG: Mapping[str, RoleCatalogEntry] = MappingProxyType(
    {
        "switch": RoleCatalogEntry(
            role_id="switch",
            display_name_ru="Коммутатор",
            synonyms=("switch", "ethernet switch", "коммутатор", "свитч"),
            behavior="hard_when_requested",
            quantity_rule="requested_device_count default 1 when clearly singular",
            validation_capabilities=(
                "port_count",
                "port_speed",
                "port_media",
                "poe",
                "l2_l3",
                "stacking",
                "airflow",
                "psu_redundancy",
            ),
        ),
        "router": RoleCatalogEntry(
            role_id="router",
            display_name_ru="Маршрутизатор",
            synonyms=("router", "маршрутизатор", "роутер"),
            behavior="hard_when_requested",
            quantity_rule="requested_device_count default 1 when clearly singular",
            validation_capabilities=("port_count", "port_speed", "port_media", "l3"),
        ),
        "firewall": RoleCatalogEntry(
            role_id="firewall",
            display_name_ru="Межсетевой экран",
            synonyms=("firewall", "ngfw", "utm", "межсетевой экран", "фаервол"),
            behavior="hard_when_requested",
            quantity_rule="requested_device_count default 1 when clearly singular",
            validation_capabilities=("port_count", "port_speed", "license", "support"),
        ),
        "access_point": RoleCatalogEntry(
            role_id="access_point",
            display_name_ru="Точка доступа",
            synonyms=("access point", "ap", "wi-fi", "wifi", "точка доступа"),
            behavior="hard_when_requested",
            quantity_rule="requested_device_count default 1 when clearly singular",
            validation_capabilities=("poe", "license", "support"),
        ),
        "transceiver": RoleCatalogEntry(
            role_id="transceiver",
            display_name_ru="Трансивер",
            synonyms=(
                "transceiver",
                "optic",
                "sfp",
                "sfp+",
                "sfp28",
                "qsfp",
                "трансивер",
                "оптика",
            ),
            behavior="hard_when_requested",
            quantity_rule="requested_optical_ports_or_explicit_quantity",
            validation_capabilities=("transceiver_form_factor", "port_speed", "port_media"),
        ),
        "dac_cable": RoleCatalogEntry(
            role_id="dac_cable",
            display_name_ru="DAC-кабель",
            synonyms=("dac", "direct attach cable", "dac cable", "dac-кабель"),
            behavior="hard_when_requested",
            quantity_rule="explicit_quantity_or_one_per_uplink_when_requested",
            validation_capabilities=("cable_type", "port_speed", "port_media"),
        ),
        "cable": RoleCatalogEntry(
            role_id="cable",
            display_name_ru="Кабель",
            synonyms=("cable", "patch cord", "aoc", "кабель", "патч-корд"),
            behavior="hard_when_requested",
            quantity_rule="explicit_quantity_or_one_per_port_when_requested",
            validation_capabilities=("cable_type", "port_speed", "port_media"),
        ),
        "license": RoleCatalogEntry(
            role_id="license",
            display_name_ru="Лицензия",
            synonyms=("license", "licence", "subscription", "лицензия", "подписка"),
            behavior="hard_when_requested",
            quantity_rule="device_count_or_explicit_quantity",
            validation_capabilities=("license", "license_required", "license_term"),
        ),
        "support": RoleCatalogEntry(
            role_id="support",
            display_name_ru="Поддержка",
            synonyms=("support", "warranty", "service", "поддержка", "гарантия", "сервис"),
            behavior="hard_when_requested",
            quantity_rule="device_count_or_explicit_quantity",
            validation_capabilities=("support", "support_term"),
        ),
        "power_supply": RoleCatalogEntry(
            role_id="power_supply",
            display_name_ru="Блок питания",
            synonyms=("psu", "power supply", "блок питания", "бп"),
            behavior="hard_when_requested",
            quantity_rule="explicit_spare_redundant_or_policy_required",
            validation_capabilities=("power_supply", "redundancy"),
        ),
        "stacking_module": RoleCatalogEntry(
            role_id="stacking_module",
            display_name_ru="Модуль стекирования",
            synonyms=("stacking module", "stack module", "стек модуль", "модуль стекирования"),
            behavior="hard_when_requested",
            quantity_rule="explicit_quantity_or_device_count_when_stacking_requires_module",
            validation_capabilities=("stacking",),
        ),
        "other_accessory": RoleCatalogEntry(
            role_id="other_accessory",
            display_name_ru="Прочий аксессуар",
            synonyms=("accessory", "option", "аксессуар", "опция"),
            behavior="optional",
            quantity_rule="requested_accessories",
            validation_capabilities=("accessory",),
        ),
    }
)


NETWORK_PRODUCT_GROUP_PROFILE = ProductGroupProfile(
    product_group_id="network",
    roles=tuple(NETWORK_ROLE_CATALOG),
    required_roles=(),
    optional_roles=tuple(
        role_id
        for role_id, role in NETWORK_ROLE_CATALOG.items()
        if role.behavior != "hard_by_default"
    ),
    role_catalog=NETWORK_ROLE_CATALOG,
    quantity_rules=MappingProxyType(
        {
            "switch": "from requested device count, default 1 if clearly singular",
            "router": "from requested device count, default 1 if clearly singular",
            "firewall": "from requested device count, default 1 if clearly singular",
            "access_point": "from requested device count, default 1 if clearly singular",
            "transceiver": "from requested optical ports or explicit quantity",
            "dac_cable": "from explicit quantity unless request says one per uplink/port",
            "cable": "from explicit quantity unless request says one per uplink/port",
            "license": "attach to device count or explicit quantity",
            "support": "attach to device count or explicit quantity",
            "power_supply": "hard only when explicit spare/redundant/separate or policy required",
            "stacking_module": "explicit quantity or device count when required for stacking",
            "other_accessory": "requested_accessories",
        }
    ),
    compatibility_dimensions=(
        "port_count",
        "port_speed",
        "port_media",
        "transceiver_form_factor",
        "poe_budget",
        "l2/l3",
        "stacking",
        "airflow",
        "psu/redundancy",
        "license/support completeness",
    ),
    equivalence_rules=(
        "repair may choose a cheaper equivalent only within the same network role eligibility",
        "switch/router/firewall alternatives must preserve hard ports, speed, media, "
        "PoE, L2/L3, stacking, airflow, PSU, stock, license and support requirements",
        "transceiver/DAC/cable alternatives must preserve form factor, speed, media, "
        "stock and requested quantity",
    ),
    no_recommendation_rules=(
        "hard port count/speed/media is not covered by a selected stocked network device",
        "hard uplink/transceiver/DAC/license/support role is missing or stock-blocked",
        "hard PoE budget, L3, stacking, airflow or redundancy requirement is not "
        "confirmed by selected components",
    ),
    commercial_output_template=MappingProxyType(
        {
            "default_output_mode": "single_best_cost_valid",
            "telegram": (
                "one copy-paste КП block",
                "network equipment line",
                "composition",
                "commercial comment",
                "mandatory engineering checklist",
            ),
            "excel_sheets": ("AI-рекомендации", "Матрица компонентов"),
        }
    ),
)


STORAGE_ROLE_CATALOG: Mapping[str, RoleCatalogEntry] = MappingProxyType(
    {
        "storage_system": RoleCatalogEntry(
            role_id="storage_system",
            display_name_ru="СХД",
            synonyms=(
                "storage system",
                "storage array",
                "san",
                "nas",
                "схд",
                "система хранения",
                "дисковый массив",
            ),
            behavior="hard_when_requested",
            quantity_rule="requested system count, default 1 when singular",
            validation_capabilities=(
                "raw_capacity_tb",
                "usable_capacity_tb",
                "redundancy_level",
                "controller_count",
                "host_protocol",
                "host_port_count",
                "host_port_speed",
                "host_port_media",
            ),
        ),
        "controller": RoleCatalogEntry(
            role_id="controller",
            display_name_ru="Контроллер СХД",
            synonyms=("controller", "storage controller", "контроллер", "контроллер схд"),
            behavior="hard_when_requested",
            quantity_rule="from explicit controller count or product bundle facts",
            validation_capabilities=("controller_count", "controller_redundancy"),
        ),
        "controller_module": RoleCatalogEntry(
            role_id="controller_module",
            display_name_ru="Контроллерный модуль",
            synonyms=("controller module", "контроллерный модуль", "модуль контроллера"),
            behavior="hard_when_requested",
            quantity_rule="from explicit controller/module count or product bundle facts",
            validation_capabilities=("controller/shelf generation",),
        ),
        "disk_shelf": RoleCatalogEntry(
            role_id="disk_shelf",
            display_name_ru="Дисковая полка",
            synonyms=("disk shelf", "drive shelf", "expansion shelf", "полка", "дисковая полка"),
            behavior="hard_when_requested",
            quantity_rule="explicit shelf count or derived only when safe",
            validation_capabilities=("shelf_count", "controller/shelf generation"),
        ),
        "drive": RoleCatalogEntry(
            role_id="drive",
            display_name_ru="Диск",
            synonyms=("drive", "disk", "накопитель", "диск"),
            behavior="hard_when_requested",
            quantity_rule=(
                "explicit drive count, or derive from capacity only when selected drive "
                "capacity and redundancy model are known"
            ),
            validation_capabilities=(
                "drive_count",
                "drive_capacity_tb",
                "drive_type",
                "drive_interface",
            ),
        ),
        "ssd": RoleCatalogEntry(
            role_id="ssd",
            display_name_ru="SSD",
            synonyms=("ssd", "solid state", "flash", "ссд", "ssd диск"),
            behavior="hard_when_requested",
            quantity_rule="explicit SSD count, or derive only when safe",
            validation_capabilities=("drive_count", "drive_capacity_tb", "drive_interface"),
        ),
        "hdd": RoleCatalogEntry(
            role_id="hdd",
            display_name_ru="HDD",
            synonyms=("hdd", "hard drive", "nl-sas", "nearline", "жесткий диск"),
            behavior="hard_when_requested",
            quantity_rule="explicit HDD count, or derive only when safe",
            validation_capabilities=("drive_count", "drive_capacity_tb", "drive_interface"),
        ),
        "cache": RoleCatalogEntry(
            role_id="cache",
            display_name_ru="Cache",
            synonyms=("cache", "кэш", "кеш"),
            behavior="hard_when_requested",
            quantity_rule="explicit cache capacity or bundle facts",
            validation_capabilities=("cache_capacity",),
        ),
        "host_port": RoleCatalogEntry(
            role_id="host_port",
            display_name_ru="Host ports",
            synonyms=("host port", "fc port", "iscsi port", "порт хоста", "порт подключения"),
            behavior="hard_when_requested",
            quantity_rule="explicit count or derived from requested ports only when safe",
            validation_capabilities=(
                "host_protocol",
                "host_port_count",
                "host_port_speed",
                "host_port_media",
            ),
        ),
        "protocol_module": RoleCatalogEntry(
            role_id="protocol_module",
            display_name_ru="Протокольный модуль",
            synonyms=(
                "protocol module",
                "fc module",
                "iscsi module",
                "nvme-of module",
                "модуль fc",
                "модуль iscsi",
            ),
            behavior="hard_when_requested",
            quantity_rule="explicit count or derived from requested protocol/ports only when safe",
            validation_capabilities=("host_protocol", "host_port_speed", "host_port_media"),
        ),
        "transceiver": RoleCatalogEntry(
            role_id="transceiver",
            display_name_ru="Трансивер",
            synonyms=("transceiver", "sfp", "sfp28", "qsfp", "optic", "трансивер", "оптика"),
            behavior="hard_when_requested",
            quantity_rule=(
                "explicit count or derived from requested host/uplink ports only when safe"
            ),
            validation_capabilities=("host_port_speed", "host_port_media"),
        ),
        "cable": RoleCatalogEntry(
            role_id="cable",
            display_name_ru="Кабель",
            synonyms=("cable", "dac", "aoc", "sas cable", "fc cable", "кабель"),
            behavior="hard_when_requested",
            quantity_rule="explicit count or derived from requested ports only when safe",
            validation_capabilities=("cable_type", "host_port_speed", "host_port_media"),
        ),
        "license": RoleCatalogEntry(
            role_id="license",
            display_name_ru="Лицензия",
            synonyms=("license", "licence", "subscription", "лицензия", "подписка"),
            behavior="hard_when_requested",
            quantity_rule="attach to system count or explicit license term",
            validation_capabilities=("license", "license_term"),
        ),
        "support": RoleCatalogEntry(
            role_id="support",
            display_name_ru="Поддержка",
            synonyms=("support", "warranty", "service", "поддержка", "гарантия", "сервис"),
            behavior="hard_when_requested",
            quantity_rule="attach to system count or explicit support term",
            validation_capabilities=("support", "support_required", "warranty_months"),
        ),
        "power_supply": RoleCatalogEntry(
            role_id="power_supply",
            display_name_ru="Блок питания",
            synonyms=("psu", "power supply", "бп", "блок питания"),
            behavior="hard_when_requested",
            quantity_rule="hard only when explicitly requested or required by selected bundle",
            validation_capabilities=("redundant_psu", "psu/redundancy"),
        ),
        "rail_kit": RoleCatalogEntry(
            role_id="rail_kit",
            display_name_ru="Рельсы",
            synonyms=("rail", "rails", "rail kit", "рельсы"),
            behavior="hard_when_requested",
            quantity_rule="hard only when explicitly requested or required by selected bundle",
            validation_capabilities=("rail_kit_required", "rack/rail completeness"),
        ),
        "other_accessory": RoleCatalogEntry(
            role_id="other_accessory",
            display_name_ru="Аксессуар",
            synonyms=("accessory", "option", "аксессуар", "опция"),
            behavior="optional",
            quantity_rule="requested_accessories",
            validation_capabilities=("accessory",),
        ),
    }
)


STORAGE_PRODUCT_GROUP_PROFILE = ProductGroupProfile(
    product_group_id="storage",
    roles=tuple(STORAGE_ROLE_CATALOG),
    required_roles=(),
    optional_roles=tuple(
        role_id
        for role_id, role in STORAGE_ROLE_CATALOG.items()
        if role.behavior != "hard_by_default"
    ),
    role_catalog=STORAGE_ROLE_CATALOG,
    quantity_rules=MappingProxyType(
        {
            "storage_system": "requested system count, default 1 when singular",
            "controller": "from explicit controller count or product bundle facts",
            "controller_module": "from explicit controller/module count or product bundle facts",
            "disk_shelf": "explicit shelf count or derived only when safe",
            "drive": (
                "explicit drive count, or derive from capacity only when selected drive "
                "capacity and redundancy model are known; otherwise no_recommendation "
                "or engineer-check required"
            ),
            "ssd": (
                "explicit SSD count, or derive from capacity only when selected SSD "
                "capacity and redundancy model are known"
            ),
            "hdd": (
                "explicit HDD count, or derive from capacity only when selected HDD "
                "capacity and redundancy model are known"
            ),
            "host_port": "explicit count or derived from requested ports only when safe",
            "protocol_module": (
                "explicit count or derived from requested protocol/ports only when safe"
            ),
            "transceiver": "explicit count or derived from requested ports only when safe",
            "cable": "explicit count or derived from requested ports only when safe",
            "license": "attach to system count or explicit support/license term",
            "support": "attach to system count or explicit support/license term",
            "power_supply": "hard only when explicitly requested or required by selected bundle",
            "rail_kit": "hard only when explicitly requested or required by selected bundle",
            "cache": "explicit cache capacity or bundle facts",
            "other_accessory": "requested_accessories",
        }
    ),
    compatibility_dimensions=(
        "controller/shelf generation",
        "drive interface",
        "drive form factor",
        "drive type",
        "raw vs usable capacity",
        "redundancy/RAID/erasure model",
        "protocol support: FC, iSCSI, NVMe-oF, SAS",
        "host port count/speed/media",
        "license/support completeness",
        "PSU/redundancy",
        "rack/rail completeness",
    ),
    equivalence_rules=(
        "repair may choose a cheaper equivalent only within the same storage role eligibility",
        "drive alternatives must preserve hard type, interface, capacity, stock and role rules",
        "controller/shelf alternatives must preserve generation and proven bundle compatibility",
        "host protocol alternatives must preserve protocol, port count, speed, media and stock",
    ),
    no_recommendation_rules=(
        "usable/raw capacity cannot be safely satisfied",
        (
            "required protocol/port/speed/media cannot be closed by selected components "
            "or proven bundle"
        ),
        "requested drive type/interface is unavailable",
        "selected drive quantity exceeds stock",
        "controller/shelf/drive generation mismatch is detected",
        "required license/support is missing and was explicit hard requirement",
        "LLM proposes component_candidate_id not in matrix",
    ),
    commercial_output_template=MappingProxyType(
        {
            "default_output_mode": "single_best_cost_valid",
            "telegram": (
                "Предварительная спецификация для КП",
                "СХД - N шт.",
                "Состав",
                "Всего к заказу",
                "Комментарий",
                "Проверить перед КП",
            ),
            "excel_sheets": ("AI-рекомендации", "Матрица компонентов"),
            "no_recommendation": "Безопасную складскую рекомендацию дать нельзя.",
            "engineering_review_required": True,
        }
    ),
)


PRODUCT_GROUP_PROFILES: Mapping[str, ProductGroupProfile] = MappingProxyType(
    {
        SERVER_PRODUCT_GROUP_PROFILE.product_group_id: SERVER_PRODUCT_GROUP_PROFILE,
        NETWORK_PRODUCT_GROUP_PROFILE.product_group_id: NETWORK_PRODUCT_GROUP_PROFILE,
        STORAGE_PRODUCT_GROUP_PROFILE.product_group_id: STORAGE_PRODUCT_GROUP_PROFILE,
    }
)


def get_product_group_profile(product_group_id: str) -> ProductGroupProfile | None:
    return PRODUCT_GROUP_PROFILES.get(product_group_id)
