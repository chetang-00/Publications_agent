import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Link, Route, Routes } from "react-router";
import { Sidebar } from "./components/Sidebar";
import { ChatRoute, NewChatPage } from "./pages/ChatPage";
import { DocumentsPage } from "./pages/DocumentsPage";

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 5_000, refetchOnWindowFocus: false, retry: 1 } },
});

function NotFound() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2">
      <p className="font-medium">Page not found.</p>
      <Link to="/" className="text-sm text-brand-600 hover:underline">
        Go to chat
      </Link>
    </div>
  );
}

export function AppRoutes() {
  return (
    <div className="flex h-full flex-col md:flex-row">
      <Sidebar />
      <main className="flex min-h-0 min-w-0 flex-1 flex-col">
        <Routes>
          <Route path="/" element={<NewChatPage />} />
          <Route path="/c/:conversationId" element={<ChatRoute />} />
          <Route path="/documents" element={<DocumentsPage />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </QueryClientProvider>
  );
}
