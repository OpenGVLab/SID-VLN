import json

def trans_soon(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        if 'trajectory' in item:
            new_traj = []
            new_item = {
                "obj_elevation": [item['trajectory']['obj_elevation']],
                "obj_heading": [item['trajectory']['obj_heading']],
                "path": []
            }
            for step in item['trajectory']['path']:
                if isinstance(step, list):
                    for vp in step:
                        new_traj.append([vp,0.0,0.0])
                else:
                    new_traj.append([step,0.0,0.0])
            new_item['path'] = new_traj
            item['trajectory'] = new_item

    merged = []
    for i in range(0, len(data), 10):
        group = data[i:i+10]
        trajectory = [item['trajectory'] for item in group]
        merged.append({
            'instr_id': i // 10,
            'trajectory': trajectory
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

def trans_rvr(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        if 'instr_id' in item:
            parts = item['instr_id'].split('_')
            item['instr_id'] = '_'.join(parts[:2])
        
        if 'target' in item:
            del item['target']
        
        if 'pred_objid' in item:
            item['predObjId'] = None if item['pred_objid'] is None else int(item['pred_objid'])
            del item['pred_objid']
        
        if 'trajectory' in item and isinstance(item['trajectory'], list):
            new_trajectory = []
            for step in item['trajectory']:
                if isinstance(step, list):
                    for vp in step:
                        new_trajectory.append([vp])
                else:
                    new_trajectory.append([step])
            item['trajectory'] = new_trajectory

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

trans_rvr('your_path/preds/submit_test_dynamic.json', 
'your_path/preds/submit_rvr.json')

trans_soon('your_path/preds/submit_test_v2.json', 
'your_path/preds/submit_soon.json')