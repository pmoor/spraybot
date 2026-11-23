pub mod camera;
pub mod canvas;
pub mod controller;
pub mod environment;
pub mod mist;
pub mod recorder;
pub mod robot_view;
pub mod state;
pub mod textures;

use bevy::prelude::*;
use state::SimStepSet;

/// Master Bevy plugin registering all SprayBot studio, robot, canvas, FX, camera, and export systems.
pub struct SimPlugin;

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            Update,
            (
                SimStepSet::InputAndClock,
                SimStepSet::RobotKinematics,
                SimStepSet::CanvasAndFx,
                SimStepSet::CameraAndGizmos,
                SimStepSet::RecordCapture,
            )
                .chain(),
        );

        app.add_plugins((
            environment::plugin,
            robot_view::plugin,
            controller::plugin,
            canvas::plugin,
            mist::plugin,
            camera::plugin,
            recorder::plugin,
        ));
    }
}
