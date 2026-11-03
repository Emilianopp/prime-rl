# Source inside a Slurm job before an eval or RL run of tinyswe-env: rootless podman on the
# node-local disk (Mila has no subuid ranges and no cgroup delegation, hence the flags) and
# the tinyswe-store image loaded into it. Exports what the env reads: TINYSWE_PODMAN,
# TINYSWE_SOCKETS.
export XDG_RUNTIME_DIR=$SLURM_TMPDIR/xdg
mkdir -p "$XDG_RUNTIME_DIR"
export TINYSWE_PODMAN="podman --root $SLURM_TMPDIR/pm/root --runroot $SLURM_TMPDIR/pm/run --storage-opt overlay.ignore_chown_errors=true --events-backend=file --cgroup-manager=cgroupfs --log-level=error"
# No session keyring per container: crun's keyrings count against the per-user kernel key
# quota (kernel.keys.maxkeys = 200 on Mila), which caps a user at ~200 live containers.
printf '[containers]\nkeyring = false\n' > "$SLURM_TMPDIR/containers.conf"
export CONTAINERS_CONF_OVERRIDE=$SLURM_TMPDIR/containers.conf
export TINYSWE_SOCKETS=$SLURM_TMPDIR/tinyswe-sockets
mkdir -p "$TINYSWE_SOCKETS"
TINYSWE_IMAGE_TAR=${TINYSWE_IMAGE_TAR:-$SCRATCH/podman-swe-test/tiny-swe/store/img/tinyswe-store.tar}
$TINYSWE_PODMAN load -q -i "$TINYSWE_IMAGE_TAR" > /dev/null 2>&1 || { echo "tinyswe image load failed: $TINYSWE_IMAGE_TAR"; exit 1; }
# More images for SWE-smith tasks (stage_smith.py): saved tars, e.g. `podman save -m` of several images.
for tar in ${TINYSWE_EXTRA_TARS:-}; do
  t0=$(date +%s)
  $TINYSWE_PODMAN load -q -i "$tar" > /dev/null 2>&1 || { echo "image load failed: $tar"; exit 1; }
  echo "loaded $tar in $(( $(date +%s) - t0 )) s"
done
echo "tinyswe images loaded on $(hostname) ($(nproc) cpus)"
