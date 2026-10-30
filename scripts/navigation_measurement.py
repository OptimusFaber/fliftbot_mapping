"""ContinuousNav SensorMeasurement wire schema, compatible with protobuf 3 and newer.

Schema source: BitBucket/fliftbot_navigation/protobuf_msgs/sensor_measurement.proto.
A private descriptor pool avoids depending on a generated module's runtime version.
"""
from google.protobuf import descriptor_pool, message_factory

DESCRIPTOR = descriptor_pool.DescriptorPool().AddSerializedFile(b'\n\x18sensor_measurement.proto"\xf1\x01\n\x11SensorMeasurement\x122\n\x0bsensor_type\x18\x01 \x01(\x0e2\x1d.SensorMeasurement.SensorType\x12\x11\n\ttimestamp\x18\x02 \x01(\x01\x12\r\n\x05value\x18\x03 \x03(\x01\x12\x10\n\x08variance\x18\x04 \x03(\x01"t\n\nSensorType\x12\x12\n\x0etop_navigation\x10\x00\x12\x1b\n\x17top_navigation_position\x10\x01\x12\x16\n\x12top_navigation_yaw\x10\x02\x12\x12\n\x0ewheel_odometry\x10\x03\x12\t\n\x05lidar\x10\x04b\x06proto3')
_message = DESCRIPTOR.message_types_by_name["SensorMeasurement"]
if hasattr(message_factory, "GetMessageClass"):
    SensorMeasurement = message_factory.GetMessageClass(_message)
else:
    SensorMeasurement = message_factory.MessageFactory().GetPrototype(_message)
