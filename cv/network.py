import asyncio

import wpilib
from aiologger import Logger
from aiologger.levels import LogLevel
from photonlibpy.targeting.multiTargetPNPResult import MultiTargetPNPResult, PnpResult
from photonlibpy.targeting.photonPipelineResult import (
    PhotonPipelineMetadata,
    PhotonPipelineResult,
)
from photonlibpy.targeting.photonTrackedTarget import PhotonTrackedTarget
from wpimath.geometry import Transform3d

from model import NetworkState, VisionSegment

log = Logger.with_default_handlers(name="photonvision_serialize", level=LogLevel.INFO)


## TODO: think about how this can be made more async
async def nt_loop(camera_state: VisionSegment, nt_state: NetworkState):
    while True:
        receive_timestamp_us = int(wpilib.Timer.getFPGATimestamp() * 1e6)
        # Handle the case where the we have a negative processing time.
        if receive_timestamp_us < camera_state.last_capture_time:
            return
        pnp_result = camera_state.last_pnp
        last_targets = camera_state.last_targets

        if pnp_result is not None:
            multi_result = MultiTargetPNPResult(
                estimatedPose=pnp_result, fiducialIDsUsed=camera_state.last_ids
            )
        else:
            multi_result = None
        metadata = PhotonPipelineMetadata(
            captureTimestampMicros=camera_state.last_capture_time,
            publishTimestampMicros=max(
                receive_timestamp_us, camera_state.last_capture_time
            ),
            sequenceID=camera_state.sequence_id,
        )
        result = PhotonPipelineResult(
            metadata=metadata,
            multitagResult=multi_result,
            targets=last_targets,
        )
        nt_state.nt_wrapper.latencyMillisEntry.set(
            result.getLatencyMillis(), receive_timestamp_us
        )

        newPacket = PhotonPipelineResult.photonStruct.pack(result)
        nt_state.nt_wrapper.rawBytesEntry.set(newPacket.getData(), receive_timestamp_us)

        hasTargets = result.hasTargets()
        nt_state.nt_wrapper.hasTargetEntry.set(hasTargets, receive_timestamp_us)
        if not hasTargets:
            nt_state.nt_wrapper.targetPitchEntry.set(0.0, receive_timestamp_us)
            nt_state.nt_wrapper.targetYawEntry.set(0.0, receive_timestamp_us)
            nt_state.nt_wrapper.targetAreaEntry.set(0.0, receive_timestamp_us)
            nt_state.nt_wrapper.targetPoseEntry.set(Transform3d(), receive_timestamp_us)
            nt_state.nt_wrapper.targetSkewEntry.set(0.0, receive_timestamp_us)
        else:
            bestTarget = result.getBestTarget()
            assert bestTarget

            nt_state.nt_wrapper.targetPitchEntry.set(
                bestTarget.getPitch(), receive_timestamp_us
            )
            nt_state.nt_wrapper.targetYawEntry.set(
                bestTarget.getYaw(), receive_timestamp_us
            )
            nt_state.nt_wrapper.targetAreaEntry.set(
                bestTarget.getArea(), receive_timestamp_us
            )
            nt_state.nt_wrapper.targetSkewEntry.set(
                bestTarget.getSkew(), receive_timestamp_us
            )

            if pnp_result is not None:
                nt_state.nt_wrapper.targetPoseEntry.set(
                    pnp_result.best, receive_timestamp_us
                )
            else:
                nt_state.nt_wrapper.targetPoseEntry.set(
                    bestTarget.getBestCameraToTarget(), receive_timestamp_us
                )

        nt_state.nt_wrapper.heartbeatPublisher.set(
            metadata.sequenceID, receive_timestamp_us
        )

        nt_state.nt_wrapper.subTable.getInstance().flush()
        await asyncio.sleep(0.0000000000000001)
