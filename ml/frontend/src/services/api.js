// API Service for TEJAS Caution Order (TSR) & Form T/409 Module

const API_BASE = "http://localhost:8000";

export async function fetchActiveCautionOrders() {
  const res = await fetch(`${API_BASE}/caution-orders/active`);
  if (!res.ok) {
    throw new Error(`Failed to fetch active caution orders: ${res.statusText}`);
  }
  return await res.json();
}

export async function fetchSections() {
  const res = await fetch(`${API_BASE}/caution-orders/sections`);
  if (!res.ok) {
    throw new Error(`Failed to fetch sections: ${res.statusText}`);
  }
  return await res.json();
}

export async function issueCautionOrder(data) {
  const res = await fetch(`${API_BASE}/caution-orders/issue`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to issue caution order: ${res.statusText}`);
  }
  return await res.json();
}

export async function revokeCautionOrder(orderId) {
  const res = await fetch(`${API_BASE}/caution-orders/${orderId}/revoke`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to revoke caution order: ${res.statusText}`);
  }
  return await res.json();
}

export function getCautionOrderPdfUrl(orderId) {
  return `${API_BASE}/caution-orders/${orderId}/pdf`;
}
