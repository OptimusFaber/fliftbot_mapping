"""Historical command compatibility and launch wiring; no robot processes started."""
import ast
import importlib.util
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from wheel_motion import resolve_odometry_model,telemetry_twist

try:
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument,ExecuteProcess
    from launch.utilities import perform_substitutions
    ROS_LAUNCH=True
except ImportError:
    ROS_LAUNCH=False


class ModelContract(unittest.TestCase):
    def test_legacy_switch_is_only_an_alias(self):
        self.assertEqual(resolve_odometry_model('', 'true'),'legacy_sin')
        self.assertEqual(resolve_odometry_model('', 'false'),'bicycle')
        self.assertEqual(resolve_odometry_model('', '', 'drive_wheel'),'drive_wheel')
        self.assertEqual(resolve_odometry_model('legacy_sin', '', 'drive_wheel'),'legacy_sin')
        with self.assertRaisesRegex(ValueError,'conflicts'):
            resolve_odometry_model('drive_wheel','true')
        with self.assertRaises(ValueError):
            resolve_odometry_model('','maybe')

    def test_distinct_physical_models(self):
        common=dict(speed_sign=-1.,steering_zero_deg=-90.,steering_sign=1.,wheelbase=1.41)
        outputs={model:telemetry_twist(-.4,-30.,model=model,**common)
                 for model in ('legacy_sin','drive_wheel','bicycle')}
        self.assertAlmostEqual(outputs['legacy_sin'][0],.4)
        self.assertAlmostEqual(outputs['drive_wheel'][0],.2)
        self.assertAlmostEqual(outputs['legacy_sin'][1],outputs['drive_wheel'][1])
        self.assertAlmostEqual(outputs['bicycle'][1],2*outputs['legacy_sin'][1])

    def test_recorded_csv_and_raw_mc_need_opposite_input_signs(self):
        # mc_dict_from_msg stores -MC.enc_lin_vel_mps in the recording CSV.
        stored_speed=.299
        raw_mc_speed=-stored_speed
        common=dict(steering_zero_deg=0.,steering_sign=1.,wheelbase=1.41)
        for model in ('legacy_sin','drive_wheel','bicycle'):
            live=telemetry_twist(raw_mc_speed,-17.746,speed_sign=-1.,model=model,**common)
            recorded=telemetry_twist(stored_speed,-17.746,speed_sign=1.,model=model,**common)
            self.assertEqual(live,recorded)

    def test_every_live_argument_is_in_readme(self):
        tree=ast.parse((ROOT/'launch/mc_lidar_localization.launch.py').read_text())
        names=[ast.literal_eval(n.args[0]) for n in ast.walk(tree)
               if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='DeclareLaunchArgument']
        readme=(ROOT/'README.md').read_text()
        for name in names:
            self.assertIn(f'| `{name}` |',readme,name)


@unittest.skipUnless(ROS_LAUNCH,'requires ROS launch Python modules')
class LaunchContract(unittest.TestCase):
    def load(self,**overrides):
        spec=importlib.util.spec_from_file_location('live_launch_contract',ROOT/'launch/mc_lidar_localization.launch.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        context=LaunchContext();context.launch_configurations.update({'mc_topic':'/fliftbot204/mc_telemetry',**overrides})
        for action in module.generate_launch_description().entities:
            if isinstance(action,DeclareLaunchArgument):
                action.execute(context)
        actions=module._nodes(context)
        commands=[[perform_substitutions(context,arg) for arg in action.cmd]
                  for action in actions if isinstance(action,ExecuteProcess)
                  and action.cmd and 'python' in perform_substitutions(context,action.cmd[0])]
        return commands

    def test_historical_command_reaches_same_components(self):
        commands=self.load(use_legacy_sin='true',wheelbase='1.41',sensor_yaw_deg='180.0',
            pose_offset_x_m='-0.04',pose_offset_y_m='-0.045',global_sampling_ratio='0.5')
        def command(name): return next(cmd for cmd in commands if any(name in s for s in cmd))
        def value(cmd,key): return cmd[cmd.index(key)+1]
        odom=command('mc_telemetry_odom_adapter.py');scan=command('direct_udp_scan_publisher.py');pose=command('live_marker98_pose.py')
        self.assertEqual(value(odom,'--model'),'legacy_sin')
        self.assertEqual(value(odom,'--wheelbase'),'1.41')
        self.assertEqual(value(scan,'--sensor-yaw-deg'),'180.0')
        self.assertEqual(value(pose,'--pose-offset-x-m'),'-0.04')
        self.assertEqual(value(pose,'--pose-offset-y-m'),'-0.045')
        self.assertNotIn('--no-send-to-navigation',pose)

    def test_scan_metadata_and_configuration_forwarding(self):
        commands=self.load(scan_rate_hz='20',configuration_basename='custom.lua',send_to_navigation='false')
        scan=next(c for c in commands if any('direct_udp_scan_publisher.py' in v for v in c))
        initial=next(c for c in commands if any('initialize_localization.py' in v for v in c))
        pose=next(c for c in commands if any('live_marker98_pose.py' in v for v in c))
        self.assertEqual(scan[scan.index('--scan-rate-hz')+1],'20')
        self.assertEqual(initial[initial.index('--configuration-basename')+1],'custom.lua')
        self.assertIn('--no-send-to-navigation',pose)

    def test_conflicting_switches_rejected_before_start(self):
        with self.assertRaises(ValueError):
            self.load(odom_model='drive_wheel',use_legacy_sin='true')


if __name__=='__main__': unittest.main()
