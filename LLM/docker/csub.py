#!/usr/bin/python3

import argparse
from datetime import datetime, timedelta
import re
import subprocess
import tempfile
import yaml

parser = argparse.ArgumentParser(description="Cluster Submit Utility")
parser.add_argument("-n", "--name", type=str, required=False,
                    help="Job name (has to be unique in the namespace)")
parser.add_argument("-c", "--command", type=str, required=False,
                    help="Command to run on the instance (default sleep for duration)")
parser.add_argument("-t", "--time", type=str, required=False,
                    help="The maximum duration allowed for this job (default 2 weeks)")
parser.add_argument("-g", "--gpus", type=int, default=1, required=False,
                    help="The number of GPUs requested (default 1)")
parser.add_argument("-i", "--image", type=str, required=False,
                    default="ic-registry.epfl.ch/mlo/pagliard-base-v2", # "pagliard-base-v2", "ic-registry.epfl.ch/mlo/pagliard-pytorch-base"
                    help="The URL of the docker image that will be used for the job")
parser.add_argument("-p", "--port", type=int, required=False,
                    help="A cluster port for connect to this node")
parser.add_argument("-u", "--user", type=str, default="user.yaml",
                    help="Path to a yaml file that defines the user")
parser.add_argument("-gt", "--gpu_type", type=str, default="G10",
                    help="type of GPU requested: G9 for V100 and G10 for A100")

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.user, 'r') as file:
        user_cfg = yaml.safe_load(file)

    if args.name is None:
        args.name = f"{user_cfg['user']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if args.time is None:
        args.time = 10 * 24 * 60 * 60
    else:
        pattern = r"((?P<days>\d+)d)?((?P<hours>\d+)h)?((?P<minutes>\d+)m)?((?P<seconds>\d+)s?)?"
        match = re.match(pattern, args.time)
        parts = {k: int(v) for k, v in match.groupdict().items() if v}
        args.time = int(timedelta(**parts).total_seconds())

    if args.command is None:
        args.command = f"sleep {args.time}"

    symlink_targets, symlink_paths = zip(*user_cfg['symlinks'].items())
    symlink_targets = ":".join(symlink_targets)
    symlink_paths = ":".join(symlink_paths)

    cfg = \
f"""
# Source: runaijob/templates/runai-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: {args.name}
  labels:
    app: runaijob
    chart: runaijob-1.0.1
    release: {args.name}
    heritage: Helm
    createdBy: "RunaiJob"
    app: runaijob
    chart: runaijob-1.0.1
    release: {args.name}
    heritage: Helm
    createdBy: "RunaiJob"
spec:
  type: 'NodePort'
  selector:
    release: {args.name}
  ports:
    - name: "22-port"
      protocol: 'TCP'
      port: 22
      targetPort: 22
{'      nodePort: '+ str(args.port) if args.port else ''}
---
# Source: runaijob/templates/runai-job.yaml
apiVersion: run.ai/v1
kind: "RunaiJob"

metadata:
  name: {args.name}
  labels:
    app: runaijob
    chart: runaijob-1.0.1
    release: {args.name}
    heritage: Helm
    createdBy: "RunaiJob"
spec:
  activeDeadlineSeconds: {args.time}
  parallelism: 1
  completions: 1
  backoffLimit: 10
  template:
    metadata:
      labels:
        app: runaijob
        chart: runaijob-1.0.1
        release: {args.name}
        heritage: Helm
        createdBy: "RunaiJob"
        user: {user_cfg['user']}
    spec:
      nodeSelector:
        run.ai/type: {args.gpu_type}
      schedulerName: runai-scheduler
      restartPolicy: Never
      hostIPC: false
      hostNetwork: false
      containers:
        - name: {args.name}
          command: [
              "/entrypoint.sh",
              "bash",
              "-c",
              "{args.command}",
          ]
          env:
            - name: HOME
              value: "/home/{user_cfg['user']}"
            - name: NB_USER
              value: {user_cfg['user']}
            - name: NB_UID
              value: "{user_cfg['uid']}"
            - name: NB_GROUP
              value: {user_cfg['group']}
            - name: NB_GID
              value: "{user_cfg['gid']}"
            - name: SYMLINK_TARGETS
              value: "{symlink_targets}"
            - name: SYMLINK_PATHS
              value: "{symlink_paths}"
          stdin:
          tty:
          image: {args.image}
          imagePullPolicy: Always
          securityContext:
            allowPrivilegeEscalation: true
          resources:
            limits:
              nvidia.com/gpu: {args.gpus}
            requests:
          volumeMounts:
            - mountPath: /scratch
              name: mlo-scratch
            # - mountPath: /home
            #   name: mlo-scratch
            #   subPath: homes
            - mountPath: /mlodata1
              name: mlodata1
            - mountPath: /mloraw1
              name: mloraw1
            - mountPath: /dev/shm  # Increase shared memory size
              name: dshm
          ports:
            - protocol: 'TCP'
              containerPort: 22
      volumes:
        - name: mlo-scratch
          persistentVolumeClaim:
            claimName: runai-mlo-pagliard-scratch
        - name: mlodata1
          persistentVolumeClaim:
            claimName: runai-mlo-pagliard-mlodata1
        - name: mloraw1
          persistentVolumeClaim:
            claimName: runai-mlo-pagliard-mloraw1
        - name: dshm  # Increase the shared memory size
          emptyDir:
            medium: Memory
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as f:
        f.write(cfg)
        f.flush()
        subprocess.run(["kubectl", "apply", "-f", f.name], check=True)

    print("\nThe following commands may come in handy:")
    print(f"runai bash {args.name} - opens an interactive shell on the pod (can ssh/su for user)")
    print(f"runai delete {args.name} - kills the job and removes it from the list of jobs")
    print(f"runai describe job {args.name} - shows information on the status/execution of the job")
    print("runai list jobs - list all jobs and their status (including ip and port for ssh)")
    print(f"runai logs {args.name} - shows the output/logs for the job")

    try:
        out = subprocess.run(
            ['bash', '-c', f'runai list jobs | grep {args.name}'],
            capture_output=True
        )
        ip, port = re.search(r"(\d{1,3}.){3}\d{1,3}:\d+", out.stdout.decode('utf-8'))[0].split(':')
        print(f"ssh {user_cfg['user']}@{ip} -p {port} - ssh into the pod (once it starts running)")
    except Exception:
        print(f"ssh {user_cfg['user']}@IP -p PORT - ssh into the pod, IP and PORT can be found with list jobs")


# command: [
#     "/entrypoint.sh",
#     "bash",
#     "-c",
#     "{args.command}",
# ]