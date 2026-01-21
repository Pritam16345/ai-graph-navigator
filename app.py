import streamlit as st
import heapq
import pandas as pd
import google.generativeai as genai
from PIL import Image

GEMINI_KEY = st.secrets.get("GEMINI_KEY", "AIzaSyBdRJEC-qP0USJh2xmTSSbC8ZqbSRdZuwg")

st.set_page_config(page_title="AI Graph Navigator", layout="wide", page_icon="🧭")

def enforce_symmetry(graph):
    """Ensures that if A connects to B, B also connects to A."""
    for node in list(graph.keys()):
        for neighbor, weight in graph[node].items():
            if neighbor not in graph: graph[neighbor] = {}
            if node not in graph[neighbor]:
                graph[neighbor][node] = weight
    return graph

def parse_graph_text(content):
    graph = {}
    cities = set()
    
    for line in content.splitlines():
        line = line.strip()
        if ':' not in line: continue
        parts = line.split(':')
        if len(parts) < 2: continue
        
        node = parts[0].strip().replace("*", "")
        cities.add(node)
        if node not in graph: graph[node] = {}
        
        neighbors_part = parts[1].strip()
        for n in neighbors_part.split(','):
            n = n.strip()
            if '(' in n and ')' in n:
                try:
                    city_str, cost_str = n.split('(')
                    neighbor = city_str.strip()
                    cost = int(cost_str.replace(')', '').strip())
                    graph[node][neighbor] = cost
                    cities.add(neighbor)
                except: continue
                    
    return enforce_symmetry(graph), sorted(list(cities))

def run_dfs(graph, start, goal):
    stack = [(start, [start], 0)]
    visited = set()
    log = []
    step = 1

    while stack:
        node, path, cost = stack.pop()
        if node in visited: continue
        visited.add(node)
        
        log.append({"Step": step, "Node": node, "Action": f"Visiting (Cost: {cost})", "Path So Far": " → ".join(path)})

        if node == goal:
            return path, cost, log

        neighbors = sorted(graph.get(node, {}).items(), key=lambda x: x[1], reverse=True)
        for neighbor, weight in neighbors:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor], cost + weight))
        step += 1
    return None, 0, log

def run_ucs(graph, start, goal):
    pq = [(0, start, [start])]
    visited = set()
    log = []
    step = 1

    while pq:
        cost, node, path = heapq.heappop(pq)
        if node in visited: continue
        visited.add(node)

        log.append({"Step": step, "Node": node, "Action": f"Visiting (Total: {cost})", "Path So Far": " → ".join(path)})

        if node == goal:
            return path, cost, log

        for neighbor, weight in graph.get(node, {}).items():
            if neighbor not in visited:
                heapq.heappush(pq, (cost + weight, neighbor, path + [neighbor]))
        step += 1
    return None, 0, log

def extract_graph_gemini(image):
    try:
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = """
        Analyze this image and extract the graph structure. 
        Identify all nodes (circles/text) and edges (lines connecting them).
        
        OUTPUT FORMAT STRICTLY:
        Node: Neighbor(Weight), Neighbor(Weight)
        
        RULES:
        1. If a number is written on a line, use it as the Weight.
        2. If no number is visible, assume Weight is 1.
        3. Ensure all connections are listed.
        4. Do not output any conversational text, just the list.
        """
        
        with st.spinner('Analysing the graph...'):
            response = model.generate_content([prompt, image])
            return response.text
    except Exception as e:
        st.error(f"Error: {e}")
        return None

st.title("🧭 AI Graph Navigator")
st.markdown("Upload **any** map and find the optimal path using AI.")

with st.sidebar:
    st.header("1. Map Source")
    uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Map Preview", use_column_width=True)
        
        if st.button("Scan Map", type="primary"):
            extracted_text = extract_graph_gemini(image)
            if extracted_text:
                st.session_state['graph_data'] = extracted_text
                st.success("Graph extracted!")
    st.markdown("---")

if 'graph_data' not in st.session_state:
    st.info("Please upload a map image and click 'Scan Map' to begin.")
else:
    col_main, col_log = st.columns([1, 1.2])

    graph, cities = parse_graph_text(st.session_state['graph_data'])

    with col_main:
        st.subheader("Search Configuration :")
        if not cities:
            st.error("No nodes found. Please check extracted data below.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                start_node = st.selectbox("Start Node", cities, index=0)
            with c2:
                goal_node = st.selectbox("Goal Node", cities, index=len(cities)-1)
            
            algo = st.radio("Select Strategy", 
                           ["DFS", 
                            "UCS"], 
                           horizontal=True)
            
            if st.button("Find Path", type="primary", use_container_width=True):
                if "DFS" in algo:
                    path, cost, log = run_dfs(graph, start_node, goal_node)
                else:
                    path, cost, log = run_ucs(graph, start_node, goal_node)
                
                st.session_state['results'] = (path, cost, log)

    with col_log:
        st.subheader("Results :")
        if 'results' in st.session_state:
            path, cost, log_data = st.session_state['results']
            
            if path:
                m1, m2 = st.columns(2)
                m1.metric("Total Cost", cost)
                m2.metric("Steps Taken", len(path)-1)
                
                st.success(f"**Path:** {' → '.join(path)}")

                with st.expander("View Detailed Execution Steps", expanded=True):
                    df_log = pd.DataFrame(log_data)
                    st.dataframe(df_log, use_container_width=True, hide_index=True)
            else:
                st.error("Goal is unreachable from start node.")

    st.divider()
    st.subheader("Graph Data Studio")
    st.markdown("Verify or edit the AI-extracted connections below.")
    
    updated_data = st.text_area(
        label="Raw Graph Data",
        value=st.session_state['graph_data'],
        height=250,
        help="Format: Node: Neighbor(Weight), Neighbor(Weight)"
    )

    if updated_data != st.session_state['graph_data']:
        st.session_state['graph_data'] = updated_data
        st.rerun()