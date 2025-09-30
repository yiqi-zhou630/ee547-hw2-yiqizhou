import argparse
import datetime
import json
import os
import sys
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError, ReadTimeoutError

ISO = "%Y-%m-%dT%H:%M:%SZ"

def utc_now_iso():
    return datetime.datetime.utcnow().strftime(ISO)


def human_mb(nbytes):
    if nbytes is None:
        return "-"
    return f"~{(nbytes/1024/1024):.1f}"


def safe_dt(o):
    try:
        return o.strftime(ISO)
    except Exception:
        return "-"


def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


def retry(fn, *, retries=1, delay=0.8, on_fail=None):
    for i in range(retries + 1):
        try:
            return fn()
        except (ReadTimeoutError, EndpointConnectionError) as ex:
            if i < retries:
                time.sleep(delay)
                continue
            if on_fail:
                on_fail(ex)
            raise


def validate_region(region):
    if not region:
        return None
    all_regions = set(boto3.session.Session().get_available_regions("ec2"))
    if region not in all_regions:
        eprint(f"ERROR Invalid region: {region}")
        sys.exit(2)
    return region


def make_session(region):
    try:
        session = boto3.session.Session(region_name=region)
        sts = session.client("sts", config=Config(retries={"max_attempts": 2}))
        ident = retry(lambda: sts.get_caller_identity(), retries=1)
        account = ident.get("Account")
        arn = ident.get("Arn")
        return session, account, arn
    except NoCredentialsError:
        eprint("ERROR No AWS credentials found. Configure with `aws configure` or env vars.")
        sys.exit(1)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in {"InvalidClientTokenId", "SignatureDoesNotMatch"}:
            eprint(f"ERROR Authentication failed: {code}")
        else:
            eprint(f"ERROR Authentication failed: {e}")
        sys.exit(1)


def list_iam_users(session):
    users = []
    try:
        iam = session.client("iam")
        paginator = iam.get_paginator("list_users")
        for page in paginator.paginate():
            for u in page.get("Users", []):
                username = u.get("UserName")
                user_arn = u.get("Arn")
                user_id = u.get("UserId")
                create_date = safe_dt(u.get("CreateDate"))
                last = "-"
                try:
                    last = safe_dt(u.get("PasswordLastUsed"))
                except Exception:
                    pass

                policies = []
                try:
                    resp = iam.list_attached_user_policies(UserName=username)
                    for p in resp.get("AttachedPolicies", []):
                        policies.append({
                            "policy_name": p.get("PolicyName"),
                            "policy_arn": p.get("PolicyArn"),
                        })
                except ClientError as e:
                    eprint(f"WARNING Access denied listing attached policies for user {username}: {e.response.get('Error', {}).get('Code')}")

                users.append({
                    "username": username,
                    "user_id": user_id,
                    "arn": user_arn,
                    "create_date": create_date,
                    "last_activity": last,
                    "attached_policies": policies,
                })
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        eprint(f"WARNING Access denied for IAM operations - skipping user enumeration ({code}).")
    return users


def list_ec2_instances(session):
    out = []
    try:
        ec2 = session.client("ec2", config=Config(retries={"max_attempts": 2}))
        paginator = ec2.get_paginator("describe_instances")
        for page in retry(lambda: paginator.paginate(), retries=1):
            for res in page.get("Reservations", []):
                for ins in res.get("Instances", []):
                    instance_id = ins.get("InstanceId")
                    itype = ins.get("InstanceType", "-")
                    state = ins.get("State", {}).get("Name", "-")
                    public_ip = ins.get("PublicIpAddress", "-")
                    private_ip = ins.get("PrivateIpAddress", "-")
                    az = ins.get("Placement", {}).get("AvailabilityZone", "-")
                    launch_time = safe_dt(ins.get("LaunchTime"))
                    image_id = ins.get("ImageId", "-")
                    sgs = [sg.get("GroupId") for sg in ins.get("SecurityGroups", [])]
                    ami_name = "-"
                    try:
                        img = ec2.describe_images(ImageIds=[image_id])
                        images = img.get("Images", [])
                        if images:
                            ami_name = images[0].get("Name", "-")
                    except ClientError:
                        pass
                    tags = {t.get("Key"): t.get("Value") for t in ins.get("Tags", [])} if ins.get("Tags") else {}

                    out.append({
                        "instance_id": instance_id,
                        "instance_type": itype,
                        "state": state,
                        "public_ip": public_ip if public_ip else "-",
                        "private_ip": private_ip if private_ip else "-",
                        "availability_zone": az,
                        "launch_time": launch_time,
                        "ami_id": image_id,
                        "ami_name": ami_name,
                        "security_groups": sgs,
                        "tags": tags,
                    })
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        eprint(f"WARNING EC2 describe failed: {code}.")
    except (ReadTimeoutError, EndpointConnectionError):
        eprint("WARNING EC2 query timed out once, skipping.")
    return out


def approx_s3_bucket_stats(session, bucket, region_hint):
    s3 = session.client("s3", region_name=region_hint)
    key_count = 0
    size_bytes = 0
    token = None
    scanned_pages = 0
    try:
        while True:
            if token:
                resp = s3.list_objects_v2(Bucket=bucket, ContinuationToken=token, MaxKeys=2000)
            else:
                resp = s3.list_objects_v2(Bucket=bucket, MaxKeys=2000)
            scanned_pages += 1
            for obj in resp.get("Contents", []):
                key_count += 1
                size_bytes += int(obj.get("Size", 0))
            if resp.get("IsTruncated") and scanned_pages < 10:
                token = resp.get("NextContinuationToken")
            else:
                break
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        eprint(f"WARNING Failed to access S3 bucket '{bucket}': {code}")
        return {"object_count": None, "size_bytes": None}
    return {"object_count": key_count, "size_bytes": size_bytes}


def list_s3_buckets(session, region):
    out = []
    try:
        s3 = session.client("s3")
        resp = retry(lambda: s3.list_buckets(), retries=1)
        for b in resp.get("Buckets", []):
            name = b.get("Name")
            created = safe_dt(b.get("CreationDate"))
            b_region = "-"
            try:
                lr = s3.get_bucket_location(Bucket=name)
                loc = lr.get("LocationConstraint")
                b_region = loc if loc else "us-east-1"
            except ClientError:
                pass
            stats = approx_s3_bucket_stats(session, name, b_region if b_region != "-" else region)
            out.append({
                "bucket_name": name,
                "creation_date": created,
                "region": b_region,
                "object_count": stats["object_count"] if stats["object_count"] is not None else "-",
                "size_bytes": stats["size_bytes"] if stats["size_bytes"] is not None else "-",
            })
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        eprint(f"WARNING S3 list failed: {code}.")
    return out


def list_security_groups(session):
    out = []
    try:
        ec2 = session.client("ec2")
        paginator = ec2.get_paginator("describe_security_groups")
        for page in paginator.paginate():
            for sg in page.get("SecurityGroups", []):
                def fmt_perm(p):
                    proto = p.get("IpProtocol", "all")
                    prange = "all"
                    if "FromPort" in p and "ToPort" in p:
                        prange = f"{p['FromPort']}-{p['ToPort']}"
                    srcs = []
                    for ip in p.get("IpRanges", []):
                        srcs.append(ip.get("CidrIp"))
                    for ip6 in p.get("Ipv6Ranges", []):
                        srcs.append(ip6.get("CidrIpv6"))
                    for gp in p.get("UserIdGroupPairs", []):
                        srcs.append(gp.get("GroupId"))
                    return {
                        "protocol": proto,
                        "port_range": prange,
                        "source": ", ".join(srcs) if srcs else "0.0.0.0/0"
                    }

                inbound = [fmt_perm(p) for p in sg.get("IpPermissions", [])]
                outbound = [fmt_perm(p) for p in sg.get("IpPermissionsEgress", [])]

                out.append({
                    "group_id": sg.get("GroupId"),
                    "group_name": sg.get("GroupName"),
                    "description": sg.get("Description", "-"),
                    "vpc_id": sg.get("VpcId", "-"),
                    "inbound_rules": inbound,
                    "outbound_rules": outbound,
                })
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        eprint(f"WARNING EC2 security group describe failed: {code}.")
    return out

def json_report(account_id, user_arn, region, iam_users, ec2_instances, s3_buckets, sec_groups):
    return {
        "account_info": {
            "account_id": account_id,
            "user_arn": user_arn,
            "region": region or boto3.session.Session().region_name or "-",
            "scan_timestamp": utc_now_iso(),
        },
        "resources": {
            "iam_users": iam_users,
            "ec2_instances": ec2_instances,
            "s3_buckets": s3_buckets,
            "security_groups": sec_groups,
        },
        "summary": {
            "total_users": len(iam_users),
            "running_instances": sum(1 for i in ec2_instances if i.get("state") == "running"),
            "total_buckets": len(s3_buckets),
            "security_groups": len(sec_groups),
        }
    }


def table_print(report) -> None:
    acct = report["account_info"]["account_id"]
    region = report["account_info"]["region"]
    ts = report["account_info"]["scan_timestamp"]
    print(f"AWS Account: {acct} ({region})")
    print(f"Scan Time: {ts}")
    print()

    us = report["resources"]["iam_users"]
    print(f"IAM USERS ({len(us)} total)")
    print(f"{'Username':20} {'Create Date':20} {'Last Activity':20} {'Policies':8}")
    for u in us:
        pol_cnt = len(u.get("attached_policies", []))
        print(f"{u['username']:20} {u['create_date']:20} {u['last_activity']:20} {pol_cnt:>8}")
    if not us:
        print("(none)")
    print()

    ec2s = report["resources"]["ec2_instances"]
    running = sum(1 for i in ec2s if i.get("state") == "running")
    stopped = sum(1 for i in ec2s if i.get("state") == "stopped")
    print(f"EC2 INSTANCES ({running} running, {stopped} stopped)")
    print(f"{'Instance ID':20} {'Type':10} {'State':8} {'Public IP':15} {'Launch Time':20}")
    for i in ec2s:
        pub = i['public_ip'] if i['public_ip'] != "-" else "-"
        print(f"{i['instance_id']:20} {i['instance_type']:10} {i['state']:8} {pub:15} {i['launch_time']:20}")
    if not ec2s:
        print("(none)")
    print()

    s3s = report["resources"]["s3_buckets"]
    print(f"S3 BUCKETS ({len(s3s)} total)")
    print(f"{'Bucket Name':28} {'Region':12} {'Created':20} {'Objects':8} {'Size (MB)':9}")
    for b in s3s:
        objs = "-" if b["object_count"] == "-" else str(b["object_count"])
        size_mb = "-" if b["size_bytes"] == "-" else human_mb(b["size_bytes"])
        print(f"{b['bucket_name']:28} {b['region']:12} {b['creation_date']:20} {objs:>8} {size_mb:>9}")
    if not s3s:
        print("(none)")
    print()

    sgs = report["resources"]["security_groups"]
    print(f"SECURITY GROUPS ({len(sgs)} total)")
    print(f"{'Group Id':16} {'Name':18} {'VPC ID':14} {'Inbound Rules':12}")
    for g in sgs:
        inb = len(g.get("inbound_rules", []))
        print(f"{g['group_id']:16} {g['group_name'][:18]:18} {g['vpc_id'][:14]:14} {inb:>12}")
    if not sgs:
        print("(none)")


def main():
    parser = argparse.ArgumentParser(description="AWS Resource Inspector")
    parser.add_argument("--region", default=None, help="AWS region to inspect (default: from config)")
    parser.add_argument("--output", default=None, help="Output file (default: stdout)")
    parser.add_argument("--format", default="json", choices=["json", "table"], help="Output format")
    args = parser.parse_args()

    region = validate_region(args.region)

    session, account_id, user_arn = make_session(region)

    iam_users = list_iam_users(session)
    ec2_instances = list_ec2_instances(session)
    s3_buckets = list_s3_buckets(session, region)
    security_groups = list_security_groups(session)

    report = json_report(account_id, user_arn, region, iam_users, ec2_instances, s3_buckets, security_groups)


    if args.format == "json":
        text = json.dumps(report, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            print(text)
    else:
        table_print(report)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        raise
    except Exception as e:
        eprint(f"ERROR Unexpected failure: {e}")
        sys.exit(1)
