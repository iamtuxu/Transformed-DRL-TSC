import xml.etree.ElementTree as ET
from xml.dom import minidom

def create_vtype(id, vclass, tau, accel, decel, sigma, length, min_gap, max_speed, car_follow_model, speed_factor, speed_dev, color=None):
    vtype = ET.Element('vType', id=id, vClass=vclass, tau=str(tau), accel=str(accel), decel=str(decel),
                       sigma=str(sigma), length=str(length), minGap=str(min_gap), maxSpeed=str(max_speed),
                       carFollowModel=car_follow_model, speedFactor=str(speed_factor), speedDev=str(speed_dev))
    if color:
        vtype.set('color', color)
    return vtype

def create_flow(id, route, begin, end, vehs_per_hour, depart_speed, depart_pos, depart_lane, type):
    flow = ET.Element('flow', id=id, route=route, begin=str(begin), end=str(end),
                      vehsPerHour=str(vehs_per_hour), departSpeed=depart_speed, departPos=depart_pos,
                      departLane=depart_lane, type=type)
    return flow

def prettify(element):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def generate_xml():
    routes = ET.Element('routes')
    # Define routes
    for direction in ['ns', 'nw', 'ne', 'we', 'wn', 'ws', 'ew', 'en', 'es', 'sn', 'se', 'sw']:
        ET.SubElement(routes, 'route', id=f"route_{direction}", edges=f"{direction[0]}_t t_{direction[1]}")

    # Define vehicle types
    car_types = {
        "customCar1": {"vclass": "passenger", "tau": 0.97, "accel": 0.73, "decel": 1.67, "sigma": 0.5, "length": 5.0,
                       "min_gap": 0.18, "max_speed": 34.1, "car_follow_model": "IDM", "speed_factor": 1.0,
                       "speed_dev": 0.1, "color": "255,1,0"},
        "customCar2": {"vclass": "passenger", "tau": 0.70, "accel": 0.73, "decel": 1.67, "sigma": 0.5, "length": 5.0,
                       "min_gap": 2.66, "max_speed": 40.0, "car_follow_model": "IDM", "speed_factor": 1.0,
                       "speed_dev": 0.1, "color": "0,255,255"},
        "customCar3": {"vclass": "passenger", "tau": 1.45, "accel": 0.73, "decel": 1.67, "sigma": 0.5, "length": 5.0,
                       "min_gap": 0.90, "max_speed": 40.0, "car_follow_model": "IDM", "speed_factor": 1.0,
                       "speed_dev": 0.1, "color": "0,0,255"},
        "customCar4": {"vclass": "passenger", "tau": 0.30, "accel": 0.73, "decel": 1.67, "sigma": 0.5, "length": 5.0,
                       "min_gap": 4.26, "max_speed": 10.0, "car_follow_model": "IDM", "speed_factor": 1.0,
                       "speed_dev": 0.1}
    }

    for car_id, attrs in car_types.items():
        routes.append(create_vtype(car_id, **attrs))

    # Define route groups
    route_groups = {
        "group1": ["ns", "sn", "nw", "se"],
        "group2": ["ne", "sw"],
        "group3": ["we", "ws", "ew", "en"],
        "group4": ["wn", "es"]
    }


    # Define flows with different vehicle types for different times and groups
    # 基础流量定义
    base_flows = {
        "customCar1NS": {"group1": 50, "group2": 25, "group3": 10, "group4": 5},
        "customCar1WE": {"group1": 10, "group2": 5, "group3": 50, "group4": 25},
        "customCar2NS": {"group1": 50, "group2": 25, "group3": 10, "group4": 5},
        "customCar2WE": {"group1": 10, "group2": 5, "group3": 50, "group4": 25},
        "customCar3NS": {"group1": 50, "group2": 25, "group3": 10, "group4": 5},
        "customCar3WE": {"group1": 10, "group2": 5, "group3": 50, "group4": 25},
        "customCar4NS": {"group1": 50, "group2": 25, "group3": 10, "group4": 5},
        "customCar4WE": {"group1": 10, "group2": 5, "group3": 50, "group4": 25}
    }

    # 乘数及其对应的NS/WE定义
    # # ## single low eval 1
    multipliers = {
        "customCar1": [(5, "NS"), (5, "WE"), (5, "NS"), (5, "WE")],
        "customCar2": [(0, "NS"), (0, "WE"), (0, "NS"), (0, "WE")],
        "customCar3": [(0, "NS"), (0, "WE"), (0., "NS"), (0, "WE")],
        "customCar4": [(0, "NS"), (0, "WE"), (0, "NS"), (0, "WE")]
    }

    # ## single medium eval 2
    multipliers = {
        "customCar1": [(0, "NS"), (0, "WE"), (0, "NS"), (0, "WE")],
        "customCar2": [(20 / 3, "NS"), (4, "WE"), (20 / 3, "NS"), (4, "WE")],
        "customCar3": [(0, "NS"), (0, "WE"), (0., "NS"), (0, "WE")],
        "customCar4": [(0, "NS"), (0, "WE"), (0, "NS"), (0, "WE")]
    }


    # ## single high eval 3
    multipliers = {
        "customCar1": [(0, "NS"), (0, "WE"), (0, "NS"), (0, "WE")],
        "customCar2": [(20 / 3, "NS"), (20 / 3, "WE"), (20 / 3, "NS"), (20 / 3, "WE")],
        "customCar3": [(0, "NS"), (0, "WE"), (0., "NS"), (0, "WE")],
        "customCar4": [(0, "NS"), (0, "WE"), (0, "NS"), (0, "WE")]
    }

    ### eval 4 various low
    multipliers = {
        "customCar1": [(5, "NS"), (0, "WE"), (5, "NS"), (0, "WE")],
        "customCar2": [(0, "NS"), (1, "WE"), (0, "NS"), (1, "WE")],
        "customCar3": [(0, "NS"), (2, "WE"), (0, "NS"), (1, "WE")],
        "customCar4": [(0, "NS"), (2, "WE"), (0, "NS"), (3, "WE")]
    }
    ######### eval 5 various medium
    multipliers = {
        "customCar1": [(5.5, "NS"), (0, "WE"), (5.5, "NS"), (0, "WE")],
        "customCar2": [(0, "NS"), (1.5, "WE"), (0, "NS"), (1.5, "WE")],
        "customCar3": [(0, "NS"), (2.5, "WE"), (0, "NS"), (1.5, "WE")],
        "customCar4": [(0, "NS"), (2.5, "WE"), (0, "NS"), (3.5, "WE")]
    }


    # # eval 6 various high
    multipliers = {
        "customCar1": [(7.5, "NS"), (0, "WE"), (7.5, "NS"), (0, "WE")],
        "customCar2": [(0, "NS"), (3, "WE"), (0, "NS"), (3, "WE")],
        "customCar3": [(0, "NS"), (3, "WE"), (0., "NS"), (1.5, "WE")],
        "customCar4": [(0, "NS"), (1.5, "WE"), (0, "NS"), (3, "WE")]
    }

    time_ranges = [(i * 1000, (i + 1) * 1000) for i in range(4)]  # 四个时间段


    flow_settings = []
    for time_range, time_index in zip(time_ranges, range(4)):
        for car_type, multipliers_list in multipliers.items():
            multiplier, ns_or_we = multipliers_list[time_index]
            if multiplier == 0:
                continue
            base_key = f"{car_type}{ns_or_we}"
            vehs_per_hour = {group: int(value * multiplier) for group, value in base_flows[base_key].items()}
            flow_settings.append({
                "time_range": time_range,
                "vehs_per_hour": vehs_per_hour,
                "car_type": car_type
            })

    # 输出结果
    for item in flow_settings:
        print(item)

    for setting in flow_settings:
        for group, routes_in_group in route_groups.items():
            for route_id in routes_in_group:
                flow_id = f"flow_{route_id}_{setting['car_type']}_{setting['time_range'][0]}"
                routes.append(create_flow(flow_id, f"route_{route_id}", setting['time_range'][0], setting['time_range'][1],
                                          setting['vehs_per_hour'][group], "max", "base", "best", setting['car_type']))

    # Write the pretty-printed XML to file
    with open('nets/2way-single-intersection/eval6.rou.xml', 'w') as file:
        file.write(prettify(routes))

generate_xml()